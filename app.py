from __future__ import annotations


import os
import sys
import cv2
import json
import torch
import tempfile
import numpy as np
from PIL import Image
from io import BytesIO
import streamlit as st
from pathlib import Path
from typing import Optional
from wiser.models.wiser import WISER
from wiser.inference.video_pool import pool_median
from wiser.data.preprocessing import haar_dwt_stack, defocus_map
from wiser.inference.calibrator import CalibrationParams, apply_calibration


try:
    from lime.lime_image import LimeImageExplainer
    from skimage.segmentation import mark_boundaries
    LIME_AVAILABLE = True
except Exception:
    LIME_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


CKPT_PATH = Path("./outputs/E04_wiser/0/ckpt/best.pt")
CALIB_PATH = Path("./outputs/E04_wiser/0/calib.json")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 256
WAVELET_SIZE = 128
MAX_VIDEO_FRAMES = 32
VIDEO_STRIDE = 5

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class SimpleFaceDetector:
    def __init__(self, image_size: int = 256, margin_frac: float = 0.30) -> None:
        self.image_size = image_size
        self.margin_frac = margin_frac
        self._haar: Optional[cv2.CascadeClassifier] = None

    def _load_haar(self) -> cv2.CascadeClassifier:
        if self._haar is None:
            self._haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        return self._haar

    def detect(self, image_rgb: np.ndarray) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        haar = self._load_haar()
        faces = haar.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(64, 64))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        margin = int(max(w, h) * self.margin_frac)
        x0 = max(0, x - margin)
        y0 = max(0, y - margin)
        x1 = min(image_rgb.shape[1], x + w + margin)
        y1 = min(image_rgb.shape[0], y + h + margin)
        crop = image_rgb[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        return cv2.resize(crop, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)


@st.cache_resource(show_spinner="Loading model …")
def load_model() -> tuple[WISER, CalibrationParams]:
    model = WISER(embed_dim=192,
                  num_classes=1,
                  dropout=0.1,
                  cdc={"in_ch": 3, "out_ch": 32, "theta": 0.7},
                  wavelet={"enabled": True, "use_defocus": True, "high_freq_gate": True},
                  bissm={"enabled": True, "d_state": 16, "d_conv": 4, "expand": 1, "bidirectional": True},
                  ssca={"enabled": True, "heads": 1, "dim": 256},
                  cnd={"enabled": True, "zc_dim": 192, "zn_dim": 64, "num_methods": 6},
                  compile=False).to(DEVICE).eval()

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True)
    state = ckpt.get("ema", ckpt.get("model", ckpt))
    model.load_state_dict(state)

    with open(CALIB_PATH, encoding="utf-8") as f:
        calib_data = json.load(f)
    calib = CalibrationParams(**calib_data)

    return model, calib


@st.cache_resource(show_spinner="Loading face detector …")
def load_detector() -> SimpleFaceDetector:
    return SimpleFaceDetector(image_size=IMAGE_SIZE, margin_frac=0.30)


def eval_transform(img: np.ndarray) -> torch.Tensor:
    if img.shape[0] != IMAGE_SIZE or img.shape[1] != IMAGE_SIZE:
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    x = img.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)


def preprocess_face_crop(crop_rgb: np.ndarray) -> dict[str, torch.Tensor]:
    rgb_t = eval_transform(crop_rgb)  # (3, 256, 256)
    wav = haar_dwt_stack(crop_rgb).astype(np.float32)  # (12, 128, 128)
    df = defocus_map(crop_rgb, target_size=WAVELET_SIZE).astype(np.float32)  # (128, 128)
    if df.ndim == 2:
        df = df[None, ...]  # (1, 128, 128)

    return {"rgb": rgb_t.unsqueeze(0).to(DEVICE),
            "wavelet": torch.from_numpy(np.ascontiguousarray(wav)).float().unsqueeze(0).to(DEVICE),
            "defocus": torch.from_numpy(np.ascontiguousarray(df)).float().unsqueeze(0).to(DEVICE)}


def predict(model: WISER, inputs: dict[str, torch.Tensor], calib: CalibrationParams) -> tuple[str, float]:
    with torch.no_grad():
        out = model(inputs["rgb"], inputs["wavelet"], inputs["defocus"])
    logits = out["logits"].float().cpu().numpy()
    probs, preds = apply_calibration(logits, calib)
    prob = float(probs[0])
    pred = int(preds[0])
    label = "FAKE" if pred == 1 else "REAL"
    confidence = prob if pred == 1 else 1.0 - prob
    return label, confidence


def lime_predict_fn_factory(model: WISER, calib: CalibrationParams):
    def _predict(images: np.ndarray) -> np.ndarray:
        images_uint8 = np.clip(images * 255, 0, 255).astype(np.uint8) # (N, H, W, 3)
        rgb_list: list[torch.Tensor] = []
        wav_list: list[torch.Tensor] = []
        df_list: list[torch.Tensor] = []

        for img in images_uint8:
            rgb_t = eval_transform(img)
            wav = haar_dwt_stack(img).astype(np.float32)
            df = defocus_map(img, target_size=WAVELET_SIZE).astype(np.float32)
            if df.ndim == 2:
                df = df[None, ...]
            rgb_list.append(rgb_t)
            wav_list.append(torch.from_numpy(np.ascontiguousarray(wav)).float())
            df_list.append(torch.from_numpy(np.ascontiguousarray(df)).float())

        rgb_batch = torch.stack(rgb_list).to(DEVICE)
        wav_batch = torch.stack(wav_list).to(DEVICE)
        df_batch = torch.stack(df_list).to(DEVICE)

        with torch.no_grad():
            out = model(rgb_batch, wav_batch, df_batch)
        logits = out["logits"].float().cpu().numpy().reshape(-1)
        probs, _ = apply_calibration(logits, calib)
        probs = probs.reshape(-1, 1)
        return np.hstack([1.0 - probs, probs]).astype(np.float64)
    return _predict


def explain_with_lime(model: WISER, calib: CalibrationParams, crop_rgb: np.ndarray, num_samples: int = 150) -> Optional[np.ndarray]:
    if not LIME_AVAILABLE:
        return None
    try:
        explainer = LimeImageExplainer()
        image_float = crop_rgb.astype(np.float32) / 255.0
        predict_fn = lime_predict_fn_factory(model, calib)
        explanation = explainer.explain_instance(image_float, predict_fn, top_labels=1, hide_color=0, num_samples=num_samples)
        top_label = explanation.top_labels[0]
        temp, mask = explanation.get_image_and_mask(top_label, positive_only=False, num_features=10, hide_rest=False)
        lime_vis = mark_boundaries(temp, mask, color=(1, 1, 0), mode="thick")
        return np.clip(lime_vis, 0, 1)
    except Exception as e:
        st.warning(f"LIME explanation failed: {e}")
        return None


def extract_frames(video_path: str | Path, max_frames: int = MAX_VIDEO_FRAMES):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            idx = 0
            yielded = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % VIDEO_STRIDE == 0:
                    yield idx, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    yielded += 1
                    if yielded >= max_frames:
                        break
                idx += 1
            return

        indices = np.linspace(0, total - 1, min(max_frames, total), dtype=int)
        for target in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(target))
            ok, frame = cap.read()
            if not ok:
                break
            yield int(target), cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def inject_courier_font():
    st.markdown(
        """
        <style>
        html, body, [class*="css"] {
            font-family: 'Courier New', Courier, monospace !important;
        }
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Courier New', Courier, monospace !important;
        }
        </style>
        """,
        unsafe_allow_html=True)


def render_results(label: str, confidence: float, extra_caption: str = "") -> str:
    pred_color = "#e74c3c" if label == "FAKE" else "#27ae60"
    caption_html = (
        f'<div style="text-align: center; font-size: 1rem; color: #777; margin-top: 1.5rem;">{extra_caption}</div>'
        if extra_caption
        else "")
    return f"""
    <div style="display: flex; flex-direction: column; justify-content: center; min-height: 260px; height: 100%;">
        <div style="text-align: center; margin-bottom: 2rem;">
            <div style="font-size: 1.1rem; color: #666; margin-bottom: 0.4rem;">Prediction</div>
            <div style="font-size: 3.5rem; font-weight: bold; color: {pred_color}; font-family: 'Courier New', Courier, monospace;">{label}</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 1.1rem; color: #666; margin-bottom: 0.4rem;">Confidence</div>
            <div style="font-size: 2.6rem; font-weight: bold; color: #333; font-family: 'Courier New', Courier, monospace;">{confidence * 100:.2f}%</div>
        </div>
        {caption_html}
    </div>
    """


def main():
    st.set_page_config(page_title="WISER", page_icon="🧙🏻‍♂️", layout="wide")
    inject_courier_font()

    st.title("️WISER Deepfake Detector 🧙🏻‍♂️")
    st.markdown("Upload an **image** or **video** and the model will predict whether it is **Real** or **Fake**")
    with st.sidebar:
        st.header("Settings")
        show_lime = st.toggle("🍋‍ LIME Explanation", value=False, help="Explain model attention via LIME (slower-inference)")
        lime_samples = 150
        if show_lime:
            lime_samples = st.slider("LIME samples", min_value=50, max_value=500, value=150, step=50,
                                     help="More samples means better explanation but slower inference")
        if not LIME_AVAILABLE and show_lime:
            st.warning("LIME not installed. Run: `pip install lime scikit-image`")

    model, calib = load_model()
    detector = load_detector()

    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None

    if st.session_state.uploaded_file is None:
        uploaded_widget = st.file_uploader("Choose a file",
                                           type=["jpg", "jpeg", "png", "bmp", "webp", "mp4", "avi", "mov", "mkv"],
                                           accept_multiple_files=False)
        if uploaded_widget is not None:
            st.session_state.uploaded_file = {"name": uploaded_widget.name,
                                              "bytes": uploaded_widget.getvalue()}
            st.rerun()
        st.info("Please upload a photo or video to get started")
        return

    file_info = st.session_state.uploaded_file
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("Upload new file"):
            st.session_state.uploaded_file = None
            st.rerun()

    suffix = Path(file_info["name"]).suffix.lower()
    is_video = suffix in {".mp4", ".avi", ".mov", ".mkv"}
    file_bytes = file_info["bytes"]

    if is_video:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
    else:
        image = Image.open(BytesIO(file_bytes)).convert("RGB")
        img_array = np.array(image)

    with st.expander("Preview", expanded=True):
        if is_video:
            st.video(file_bytes)
        else:
            st.image(img_array, caption="Uploaded image", use_container_width="auto")

    if st.button("Run Detection", type="primary"):
        with st.spinner("Detecting face & running inference …"):
            if is_video:
                all_logits: list[float] = []
                valid_frames = 0
                sample_crop: Optional[np.ndarray] = None
                sample_idx: int = 0

                for idx, rgb in extract_frames(tmp_path, max_frames=MAX_VIDEO_FRAMES):
                    crop = detector.detect(rgb)
                    if crop is None:
                        continue
                    valid_frames += 1
                    if sample_crop is None:
                        sample_crop = crop
                        sample_idx = idx

                    inputs = preprocess_face_crop(crop)
                    with torch.no_grad():
                        out = model(inputs["rgb"], inputs["wavelet"], inputs["defocus"])
                    logit = float(out["logits"].float().cpu().numpy()[0])
                    all_logits.append(logit)

                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

                if valid_frames == 0:
                    st.error("No faces detected in any sampled frame. Please try another video")
                    return

                cal_probs = []
                for logit in all_logits:
                    probs, _ = apply_calibration(np.array([logit], dtype=np.float32), calib)
                    cal_probs.append(float(probs[0]))
                pooled_prob = pool_median(np.array(cal_probs, dtype=np.float32))
                pred = 1 if pooled_prob >= float(calib.threshold) else 0
                label = "FAKE" if pred == 1 else "REAL"
                confidence = pooled_prob if pred == 1 else 1.0 - pooled_prob

                if show_lime and LIME_AVAILABLE and sample_crop is not None:
                    with st.spinner("Generating LIME explanation (this may take a moment) …"):
                        lime_img = explain_with_lime(model, calib, sample_crop, num_samples=lime_samples)

                    cols = st.columns([1, 1, 1])
                    with cols[0]:
                        st.subheader("Detected face")
                        if sample_crop is not None:
                            st.image(sample_crop, caption=f"Frame {sample_idx}", width=280)
                    with cols[1]:
                        st.subheader("LIME explanation")
                        if lime_img is not None:
                            st.image(
                                lime_img,
                                caption="Yellow borders -> important regions",
                                width=280,
                            )
                        else:
                            st.info("Could not generate LIME explanation")
                    with cols[2]:
                        st.markdown(render_results(label, confidence,
                                                   extra_caption=f"Frames analyzed: {valid_frames}  |  Pooling: median"), unsafe_allow_html=True)
                else:
                    left_col, right_col = st.columns([1, 1])
                    with left_col:
                        st.subheader("Detected face")
                        if sample_crop is not None:
                            st.image(sample_crop, caption=f"Frame {sample_idx}", width=280)
                    with right_col:
                        st.markdown(render_results(label, confidence,
                                                   extra_caption=f"Frames analyzed: {valid_frames}  |  Pooling: median"), unsafe_allow_html=True)

            else:
                crop = detector.detect(img_array)
                if crop is None:
                    st.error("No face detected. Please try another image")
                    return

                inputs = preprocess_face_crop(crop)
                label, confidence = predict(model, inputs, calib)

                if show_lime and LIME_AVAILABLE:
                    with st.spinner("Generating LIME explanation (this may take a moment) …"):
                        lime_img = explain_with_lime(model, calib, crop, num_samples=lime_samples)

                    cols = st.columns([1, 1, 1])
                    with cols[0]:
                        st.subheader("Detected face")
                        st.image(crop, caption="Face crop used for inference", width=280)
                    with cols[1]:
                        st.subheader("LIME explanation")
                        if lime_img is not None:
                            st.image(lime_img, caption="Yellow borders -> important regions", width=280)
                        else:
                            st.info("Could not generate LIME explanation")
                    with cols[2]:
                        st.markdown(render_results(label, confidence), unsafe_allow_html=True)
                else:
                    left_col, right_col = st.columns([1, 1])
                    with left_col:
                        st.subheader("Detected face")
                        st.image(crop, caption="Face crop used for inference", width=280)
                    with right_col:
                        st.markdown(render_results(label, confidence), unsafe_allow_html=True)

        with st.expander("Advanced details"):
            st.json({"device": str(DEVICE), "calibration_temperature": round(calib.temperature, 4),
                     "calibration_threshold": round(calib.threshold, 4), "calibration_prior_bias": round(calib.prior_bias, 4)})


if __name__ == "__main__":
    main()