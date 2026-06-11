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


CKPT_PATH = Path("./outputs/E14_wiser_rf_full/0/ckpt/best.pt")
CALIB_PATH = Path("./outputs/E14_wiser_rf_full/0/calib.json")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 256
WAVELET_SIZE = 128
MAX_VIDEO_FRAMES = 32
VIDEO_STRIDE = 5

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

METHOD_NAMES = ["real", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"]


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


def normalize_vis(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - float(np.nanmin(x))
    denom = float(np.nanmax(x)) + eps
    return np.clip(x / denom, 0.0, 1.0)


def display_width(img, max_width: int = 512) -> int:
    """Return a display width for UI rendering (75 % of original, capped)."""
    if isinstance(img, np.ndarray) and img.ndim >= 2:
        return min(int(img.shape[1] * 0.75), max_width)
    if hasattr(img, "size"):
        return min(int(img.size[0] * 0.75), max_width)
    return min(220, max_width)


def overlay_heatmap(crop_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heatmap = normalize_vis(heatmap)
    heatmap_uint8 = np.uint8(255 * heatmap)
    color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    color = cv2.resize(color, (crop_rgb.shape[1], crop_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    out = (1.0 - alpha) * crop_rgb.astype(np.float32) + alpha * color.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def compute_wavelet_visuals(crop_rgb: np.ndarray) -> dict[str, np.ndarray]:
    wav = haar_dwt_stack(crop_rgb).astype(np.float32)  # (12, 128, 128)

    ll = wav[[0, 4, 8]].mean(axis=0)
    lh = np.abs(wav[[1, 5, 9]]).mean(axis=0)
    hl = np.abs(wav[[2, 6, 10]]).mean(axis=0)
    hh = np.abs(wav[[3, 7, 11]]).mean(axis=0)
    hf = lh + hl + hh

    return {
        "Wavelet LL / structure": normalize_vis(ll),
        "Wavelet LH / horizontal-detail energy": normalize_vis(lh),
        "Wavelet HL / vertical-detail energy": normalize_vis(hl),
        "Wavelet HH / diagonal-texture energy": normalize_vis(hh),
        "High-frequency aggregate": normalize_vis(hf),
    }


def compute_defocus_visual(crop_rgb: np.ndarray) -> np.ndarray:
    df = defocus_map(crop_rgb, target_size=WAVELET_SIZE).astype(np.float32)
    if df.ndim == 3:
        df = df.squeeze()
    return normalize_vis(df)


def predict_detailed(model: WISER, inputs: dict[str, torch.Tensor],
                     calib: CalibrationParams, *, return_features: bool = False) -> dict[str, object]:
    with torch.no_grad():
        out = model(inputs["rgb"],
                    inputs["wavelet"],
                    inputs["defocus"],
                    return_features=return_features)

    logits = out["logits"].float().detach().cpu().numpy().reshape(-1)
    probs, preds = apply_calibration(logits, calib)
    fake_prob = float(probs[0])
    pred = int(preds[0])
    label = "FAKE" if pred == 1 else "REAL"
    confidence = fake_prob if pred == 1 else 1.0 - fake_prob

    return {
        "label": label,
        "confidence": confidence,
        "fake_probability": fake_prob,
        "threshold": float(calib.threshold),
        "logit": float(logits[0]),
        "model_output": out,
    }


def predict(model: WISER, inputs: dict[str, torch.Tensor], calib: CalibrationParams) -> tuple[str, float]:
    result = predict_detailed(model, inputs, calib)
    return str(result["label"]), float(result["confidence"])


def clone_inputs(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in inputs.items()}


def make_ablation_inputs(inputs: dict[str, torch.Tensor], mode: str) -> dict[str, torch.Tensor]:
    ablated = clone_inputs(inputs)

    if mode == "no_defocus":
        ablated["defocus"].zero_()

    elif mode == "no_wavelet":
        ablated["wavelet"].zero_()

    elif mode == "no_high_frequency":
        keep = {0, 4, 8}
        for ch in range(ablated["wavelet"].shape[1]):
            if ch not in keep:
                ablated["wavelet"][:, ch].zero_()

    elif mode == "rgb_spatial_mean":
        rgb = ablated["rgb"]
        mean = rgb.mean(dim=(2, 3), keepdim=True)
        ablated["rgb"] = mean.expand_as(rgb).clone()

    return ablated


def compute_ablation_report(model: WISER, inputs: dict[str, torch.Tensor], calib: CalibrationParams) -> list[dict[str, object]]:
    modes = [("Full model", "full"),
             ("No defocus", "no_defocus"),
             ("No wavelet", "no_wavelet"),
             ("No high-frequency wavelet bands", "no_high_frequency"),
             ("RGB spatial mean", "rgb_spatial_mean")]

    rows = []
    base_prob = None

    for label, mode in modes:
        ab_inputs = inputs if mode == "full" else make_ablation_inputs(inputs, mode)
        result = predict_detailed(model, ab_inputs, calib)
        fake_prob = float(result["fake_probability"])
        if base_prob is None:
            base_prob = fake_prob
        rows.append({
            "variant": label,
            "fake_probability": fake_prob,
            "delta_vs_full": fake_prob - base_prob,
        })

    return rows


def render_ablation_report(rows: list[dict[str, object]]) -> None:
    st.subheader("Decision factor ablation")
    st.caption("Ablation estimates how the calibrated fake probability changes when a stream is suppressed. "
               "Large drops suggest that the suppressed stream contributed to the FAKE decision.")
    st.dataframe(rows, use_container_width=True)
    st.bar_chart(rows, x="variant", y="fake_probability")


def compute_fused_gradcam(model: WISER, inputs: dict[str, torch.Tensor], target: str = "predicted",) -> np.ndarray:
    model.zero_grad(set_to_none=True)

    rgb = inputs["rgb"].detach().clone().requires_grad_(True)
    wav = inputs["wavelet"].detach().clone()
    df = inputs["defocus"].detach().clone()

    out = model(rgb, wav, df, return_features=True)
    logits = out["logits"]
    fused = out["fused"]

    logit = logits[0]
    if target == "fake":
        score = logit
    elif target == "real":
        score = -logit
    else:
        score = logit if float(logit.detach().cpu()) >= 0 else -logit

    grads = torch.autograd.grad(score, fused, retain_graph=False, create_graph=False)[0]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * fused).sum(dim=1))[0]

    cam_np = cam.detach().cpu().numpy()
    cam_np = cv2.resize(cam_np, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    return normalize_vis(cam_np)


def render_gradcam(crop_rgb: np.ndarray, cam: np.ndarray) -> None:
    overlay = overlay_heatmap(crop_rgb, cam)
    col1, col2 = st.columns(2)
    with col1:
        st.image(cam, caption="Fused Grad-CAM heatmap", width=384)
    with col2:
        st.image(overlay, caption="Fused Grad-CAM overlay", width=384)


def attention_to_map(attn: torch.Tensor, size: int = IMAGE_SIZE) -> np.ndarray:
    a = attn[0].mean(dim=0)
    n = int(a.numel())
    side = int(round(n ** 0.5))
    a = a.reshape(side, side).detach().cpu().numpy()
    a = cv2.resize(a, (size, size), interpolation=cv2.INTER_LINEAR)
    return normalize_vis(a)


def render_ssca_attention(crop_rgb: np.ndarray, model: WISER) -> None:
    attn_rw = getattr(model.ssca, "last_attn_rw", None)
    attn_wr = getattr(model.ssca, "last_attn_wr", None)
    gate = getattr(model.ssca, "last_gate", None)

    if attn_rw is None or attn_wr is None:
        st.info("SSCA attention tensors are not available. Add last_attn_* storage in wiser/models/ssca.py.")
        return

    rw_map = attention_to_map(attn_rw)
    wr_map = attention_to_map(attn_wr)

    rw_overlay = overlay_heatmap(crop_rgb, rw_map)
    wr_overlay = overlay_heatmap(crop_rgb, wr_map)

    col1, col2 = st.columns(2)
    with col1:
        st.image(rw_overlay, caption="RGB → Wavelet attention", width=384)
    with col2:
        st.image(wr_overlay, caption="Wavelet → RGB attention", width=384)

    if gate is not None:
        gate_mean = float(gate.mean().detach().cpu())
        st.metric("Mean SSCA RGB gate", f"{gate_mean:.3f}")
        st.caption("Higher values mean the fused representation weighted the RGB branch more strongly; lower values mean stronger wavelet-stream weighting.")


def render_cnd_report(model_output: dict[str, torch.Tensor]) -> None:
    if "z_c" not in model_output or "z_n" not in model_output:
        st.info("CND outputs are not available for this checkpoint/config.")
        return

    st.subheader("CND decision report")
    st.caption("CND separates manipulation evidence from method-dependent nuisance evidence. "
               "The real/fake classifier uses z_c; method logits are diagnostic only.")

    zc_norm = float(model_output["z_c"].norm(dim=-1).mean().detach().cpu())
    zn_norm = float(model_output["z_n"].norm(dim=-1).mean().detach().cpu())

    col1, col2 = st.columns(2)
    with col1:
        st.metric("z_c norm", f"{zc_norm:.3f}")
    with col2:
        st.metric("z_n norm", f"{zn_norm:.3f}")

    method_logits = model_output.get("method_logits_zn")
    if method_logits is not None:
        probs = torch.softmax(method_logits[0], dim=0).detach().cpu().numpy()
        rows = [
            {"method": METHOD_NAMES[i], "probability": float(p)}
            for i, p in enumerate(probs)
        ]
        st.dataframe(rows, use_container_width=True)
        st.bar_chart(rows, x="method", y="probability")


def build_explanation_summary(result: dict[str, object], ablation_rows: list[dict[str, object]] | None = None,
                              frame_records: list[dict[str, object]] | None = None,) -> str:
    label = str(result["label"])
    fake_prob = float(result["fake_probability"])
    threshold = float(result["threshold"])

    parts = [f"The calibrated fake probability is {fake_prob:.3f} with threshold {threshold:.3f}, so the model predicts {label}.",]

    if ablation_rows:
        full = float(ablation_rows[0]["fake_probability"])
        drops = [
            (row["variant"], full - float(row["fake_probability"]))
            for row in ablation_rows[1:]
        ]
        drops = sorted(drops, key=lambda x: x[1], reverse=True)
        if drops and drops[0][1] > 0:
            parts.append(f"The largest probability drop came from suppressing: {drops[0][0]}.")

    if frame_records:
        above = sum(float(r["fake_probability"]) >= threshold for r in frame_records)
        total = len(frame_records)
        parts.append(f"For the video, {above}/{total} analyzed face frames were above the calibrated threshold.")

    parts.append("Treat this as model evidence, not forensic ground truth.")
    return " ".join(parts)


def render_visual_evidence(crop_rgb: np.ndarray) -> None:
    st.subheader("Visual forensic evidence")
    st.caption("These maps show the actual model inputs and low-level forensic cues. "
               "They are evidence views, not independent proof of manipulation.")

    df_vis = compute_defocus_visual(crop_rgb)
    wav_vis = compute_wavelet_visuals(crop_rgb)

    rows = [("Face crop", crop_rgb),
            ("Defocus map", df_vis),
            *list(wav_vis.items()),]

    cols = st.columns(3)
    for i, (name, img) in enumerate(rows):
        with cols[i % 3]:
            st.image(img, caption=name, width=256)

def collect_video_frame_records(video_path: str | Path,
                                model: WISER,
                                detector: SimpleFaceDetector,
                                calib: CalibrationParams,) -> list[dict[str, object]]:
    frame_records: list[dict[str, object]] = []
    for idx, rgb in extract_frames(video_path, max_frames=MAX_VIDEO_FRAMES):
        crop = detector.detect(rgb)
        if crop is None:
            continue

        inputs = preprocess_face_crop(crop)
        with torch.no_grad():
            out = model(inputs["rgb"], inputs["wavelet"], inputs["defocus"])
        logit = float(out["logits"].float().cpu().numpy()[0])
        probs, _ = apply_calibration(np.array([logit], dtype=np.float32), calib)
        fake_prob = float(probs[0])

        frame_records.append({"frame_idx": int(idx),
                              "crop": crop,
                              "logit": logit,
                              "fake_probability": fake_prob,})
    return frame_records


def render_video_timeline(frame_records: list[dict[str, object]], calib: CalibrationParams) -> None:
    if not frame_records:
        st.info("No valid face frames available for timeline.")
        return

    rows = [
        {
            "frame": int(r["frame_idx"]),
            "fake_probability": float(r["fake_probability"]),
            "threshold": float(calib.threshold),
        }
        for r in frame_records
    ]
    st.line_chart(rows, x="frame", y=["fake_probability", "threshold"])

    top = sorted(frame_records, key=lambda r: float(r["fake_probability"]), reverse=True)[:3]
    cols = st.columns(len(top))
    for col, r in zip(cols, top):
        with col:
            st.image(
                r["crop"],
                caption=f"Frame {r['frame_idx']} | fake p={float(r['fake_probability']):.3f}",
                width=display_width(r["crop"]),
            )

def lime_predict_fn_factory(model: WISER, calib: CalibrationParams):
    def _predict(images: np.ndarray) -> np.ndarray:
        images_uint8 = np.clip(images * 255, 0, 255).astype(np.uint8)
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


def render_prediction_tab(crop_rgb: np.ndarray, result: dict[str, object],
                          ablation_rows: list[dict[str, object]] | None = None, frame_records: list[dict[str, object]] | None = None,
                          is_video: bool = False,) -> None:
    st.subheader("Detected face")
    st.image(crop_rgb, caption="Face crop used for inference", width=display_width(crop_rgb))


def render_model_explanations_tab(crop_rgb: np.ndarray, model: WISER,
                                  inputs: dict[str, torch.Tensor], calib: CalibrationParams, show_lime: bool,
                                  lime_samples: int, show_gradcam: bool, show_attention: bool,) -> None:
    has_lime = False
    if show_lime and LIME_AVAILABLE:
        with st.spinner("Generating LIME explanation (this may take a moment) …"):
            lime_img = explain_with_lime(model, calib, crop_rgb, num_samples=lime_samples)
        if lime_img is not None:
            has_lime = True
            st.image(lime_img, caption="LIME superpixel explanation", width=384)
            st.caption(
                "LIME perturbs image regions and fits a local interpretable approximation around this prediction. "
                "Yellow boundaries mark influential superpixels, not necessarily manipulated pixels."
            )
        else:
            st.info("Could not generate LIME explanation")
    elif show_lime and not LIME_AVAILABLE:
        st.warning("LIME not installed. Run: `pip install lime scikit-image`")

    if show_gradcam:
        try:
            cam = compute_fused_gradcam(model, inputs)
            render_gradcam(crop_rgb, cam)
        except Exception as e:
            st.warning(f"Grad-CAM failed: {e}")

    if show_attention:
        render_ssca_attention(crop_rgb, model)

    if not show_lime and not show_gradcam and not show_attention:
        st.info("Enable an explanation toggle in the sidebar to see model explanations.")


def render_decision_factors_tab(model: WISER, inputs: dict[str, torch.Tensor],
                                calib: CalibrationParams,
                                show_ablation: bool, model_output: dict[str, torch.Tensor] | None,) -> None:
    if show_ablation:
        with st.spinner("Running ablation passes …"):
            ab_rows = compute_ablation_report(model, inputs, calib)
        render_ablation_report(ab_rows)
    else:
        st.info("Enable 'Decision factor ablation' in the sidebar to see which input streams affect the prediction.")

    if model_output is not None:
        gate = getattr(model.ssca, "last_gate", None)
        if gate is not None:
            gate_mean = float(gate.mean().detach().cpu())
            st.metric("Mean SSCA RGB gate", f"{gate_mean:.3f}")
            st.caption("Higher values mean the fused representation weighted the RGB branch more strongly; "
                       "lower values mean stronger wavelet-stream weighting.")
        render_cnd_report(model_output)


def render_advanced_details_tab(result: dict[str, object], calib: CalibrationParams, explanation_settings: dict[str, object],
                                frame_records: list[dict[str, object]] | None = None,) -> None:
    details = {"device": str(DEVICE),
               "calibration_temperature": round(calib.temperature, 4),
               "calibration_threshold": round(calib.threshold, 4),
               "calibration_prior_bias": round(calib.prior_bias, 4),
               "frames_analyzed": len(frame_records) if frame_records else 1,
               "explanation_settings": explanation_settings,}
    st.json(details)


def main():
    st.set_page_config(page_title="WISER", page_icon="🧙🏻‍♂️", layout="wide")
    inject_courier_font()

    st.title("️WISER Deepfake Detector 🧙🏻‍♂️")
    st.markdown("Upload an **image** or **video** and the model will predict whether it is **Real** or **Fake**")

    with st.sidebar:
        st.header("Settings")

        st.subheader("Explanations")
        show_visual_evidence = st.toggle("Visual evidence board", value=True)
        show_lime = st.toggle("LIME explanation", value=False, help="Slower; uses image perturbations.")
        show_gradcam = st.toggle("Fused Grad-CAM", value=False, help="Uses model gradients; slower.")
        show_ablation = st.toggle("Decision factor ablation", value=False, help="Runs extra inference passes.")
        show_cnd = st.toggle("CND report", value=True)
        show_attention = st.toggle("SSCA attention", value=False, help="Requires SSCA attention storage patch.")

        lime_samples = 150
        if show_lime:
            lime_samples = st.slider("LIME samples", min_value=50, max_value=500, value=150, step=50)
        if not LIME_AVAILABLE and show_lime:
            st.warning("LIME not installed. Run: `pip install lime scikit-image`")

    model, calib = load_model()
    detector = load_detector()

    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None

    if st.session_state.uploaded_file is None:
        uploaded_widget = st.file_uploader(
            "Choose a file",
            type=["jpg", "jpeg", "png", "bmp", "webp", "mp4", "avi", "mov", "mkv"],
            accept_multiple_files=False,
        )
        if uploaded_widget is not None:
            st.session_state.uploaded_file = {"name": uploaded_widget.name, "bytes": uploaded_widget.getvalue()}
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

    video_width = None
    if is_video:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        _cap = cv2.VideoCapture(tmp_path)
        if _cap.isOpened():
            video_width = int(_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            _cap.release()
    else:
        image = Image.open(BytesIO(file_bytes)).convert("RGB")
        img_array = np.array(image)

    with st.expander("Preview", expanded=True):
        if is_video:
            st.video(file_bytes, width=video_width // 4 if video_width else None)
        else:
            st.image(img_array, caption="Uploaded image", width=display_width(img_array))

    if st.button("Run Detection", type="primary"):
        with st.spinner("Detecting face & running inference …"):
            if is_video:
                frame_records = collect_video_frame_records(tmp_path, model, detector, calib)

                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

                if not frame_records:
                    st.error("No faces detected in any sampled frame. Please try another video")
                    return

                cal_probs = np.array([r["fake_probability"] for r in frame_records], dtype=np.float32)
                pooled_prob = pool_median(cal_probs)
                pred = 1 if pooled_prob >= float(calib.threshold) else 0
                label = "FAKE" if pred == 1 else "REAL"
                confidence = pooled_prob if pred == 1 else 1.0 - pooled_prob

                result = {
                    "label": label,
                    "confidence": confidence,
                    "fake_probability": pooled_prob,
                    "threshold": float(calib.threshold),
                    "logit": float("nan"),
                    "model_output": None,
                }

                most_suspicious = max(frame_records, key=lambda r: float(r["fake_probability"]))
                explain_crop = most_suspicious["crop"]
                explain_inputs = preprocess_face_crop(explain_crop)

                # Run detailed prediction on the most suspicious frame for feature extraction
                if show_cnd or show_gradcam or show_attention:
                    detailed = predict_detailed(model, explain_inputs, calib, return_features=True)
                    result["model_output"] = detailed["model_output"]

                ablation_rows = None
                if show_ablation:
                    ablation_rows = compute_ablation_report(model, explain_inputs, calib)

                cam = None
                if show_gradcam:
                    try:
                        cam = compute_fused_gradcam(model, explain_inputs)
                    except Exception as e:
                        st.warning(f"Grad-CAM failed: {e}")

                # Top-level result panel next to Preview
                st.subheader("Detection Result")
                res_col1, res_col2 = st.columns([3, 2])
                with res_col1:
                    st.video(file_bytes, width=video_width // 4 if video_width else None)
                with res_col2:
                    _extra = f"Frames analyzed: {len(frame_records)}  |  Pooling: median" if frame_records else ""
                    st.markdown(render_results(result["label"], result["confidence"], extra_caption=_extra), unsafe_allow_html=True)
                    st.metric("Calibrated fake probability", f"{float(result['fake_probability']):.4f}")
                    st.metric("Decision threshold", f"{float(result['threshold']):.4f}")

                _summary = build_explanation_summary(result, ablation_rows=ablation_rows, frame_records=frame_records)
                st.markdown(
                    f'<div style="text-align: center; font-size: 1.25rem; line-height: 1.6; padding: 0.75rem 0;">{_summary}</div>',
                    unsafe_allow_html=True,
                )

                tabs = st.tabs(["Prediction", "Visual Evidence", "Model Explanation", "Decision Factors", "Video Timeline", "Advanced Details"])

                with tabs[0]:
                    render_prediction_tab(
                        explain_crop,
                        result,
                        ablation_rows=ablation_rows,
                        frame_records=frame_records,
                        is_video=True,
                    )

                with tabs[1]:
                    if show_visual_evidence:
                        render_visual_evidence(explain_crop)
                    else:
                        st.info("Enable 'Visual evidence board' in the sidebar to see forensic cues.")

                with tabs[2]:
                    render_model_explanations_tab(
                        explain_crop,
                        model,
                        explain_inputs,
                        calib,
                        show_lime=show_lime,
                        lime_samples=lime_samples,
                        show_gradcam=show_gradcam,
                        show_attention=show_attention,
                    )

                with tabs[3]:
                    render_decision_factors_tab(
                        model,
                        explain_inputs,
                        calib,
                        show_ablation=show_ablation,
                        model_output=result.get("model_output"),
                    )

                with tabs[4]:
                    render_video_timeline(frame_records, calib)

                with tabs[5]:
                    render_advanced_details_tab(
                        result,
                        calib,
                        explanation_settings={
                            "show_visual_evidence": show_visual_evidence,
                            "show_lime": show_lime,
                            "show_gradcam": show_gradcam,
                            "show_ablation": show_ablation,
                            "show_cnd": show_cnd,
                            "show_attention": show_attention,
                            "lime_samples": lime_samples,
                        },
                        frame_records=frame_records,
                    )

            else:
                crop = detector.detect(img_array)
                if crop is None:
                    st.error("No face detected. Please try another image")
                    return

                inputs = preprocess_face_crop(crop)
                result = predict_detailed(model, inputs, calib, return_features=show_cnd)
                model_output = result["model_output"]

                ablation_rows = None
                if show_ablation:
                    ablation_rows = compute_ablation_report(model, inputs, calib)

                cam = None
                if show_gradcam:
                    try:
                        cam = compute_fused_gradcam(model, inputs)
                    except Exception as e:
                        st.warning(f"Grad-CAM failed: {e}")

                # Top-level result panel next to Preview
                st.subheader("Detection Result")
                res_col1, res_col2 = st.columns([3, 2])
                with res_col1:
                    st.image(img_array, caption="Uploaded image", width=display_width(img_array))
                with res_col2:
                    st.markdown(render_results(result["label"], result["confidence"]), unsafe_allow_html=True)
                    st.metric("Calibrated fake probability", f"{float(result['fake_probability']):.4f}")
                    st.metric("Decision threshold", f"{float(result['threshold']):.4f}")

                _summary = build_explanation_summary(result, ablation_rows=ablation_rows)
                st.markdown(
                    f'<div style="text-align: center; font-size: 1.25rem; line-height: 1.6; padding: 0.75rem 0;">{_summary}</div>',
                    unsafe_allow_html=True,
                )

                tabs = st.tabs(["Prediction", "Visual Evidence", "Model Explanation", "Decision Factors", "Advanced Details"])

                with tabs[0]:
                    render_prediction_tab(crop, result, ablation_rows=ablation_rows)

                with tabs[1]:
                    if show_visual_evidence:
                        render_visual_evidence(crop)
                    else:
                        st.info("Enable 'Visual evidence board' in the sidebar to see forensic cues.")

                with tabs[2]:
                    render_model_explanations_tab(
                        crop,
                        model,
                        inputs,
                        calib,
                        show_lime=show_lime,
                        lime_samples=lime_samples,
                        show_gradcam=show_gradcam,
                        show_attention=show_attention,
                    )

                with tabs[3]:
                    render_decision_factors_tab(
                        model,
                        inputs,
                        calib,
                        show_ablation=show_ablation,
                        model_output=model_output,
                    )

                with tabs[4]:
                    render_advanced_details_tab(
                        result,
                        calib,
                        explanation_settings={
                            "show_visual_evidence": show_visual_evidence,
                            "show_lime": show_lime,
                            "show_gradcam": show_gradcam,
                            "show_ablation": show_ablation,
                            "show_cnd": show_cnd,
                            "show_attention": show_attention,
                            "lime_samples": lime_samples,
                        },
                    )


if __name__ == "__main__":
    main()