from __future__ import annotations


import sys
import cv2
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from wiser.utils.logging import get_logger, setup_logging
from wiser.visualization import architecture_svg as arch
from wiser.visualization import ablation_plot, calibration, confusion_matrix
from wiser.visualization import frequency_spectrum, param_efficiency, robustness_curves
from wiser.visualization import roc_pr, training_curves, tsne_umap
from wiser.visualization import (cnd_disentanglement, prototype_geometry, real_manifold, recall_grid, spectrum_comparison)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def _runs_for_curves(runs_root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for exp_dir in sorted(runs_root.iterdir()):
        if not exp_dir.is_dir():
            continue
        zero = exp_dir / "0"
        if (zero / "predictions.npz").exists():
            out[exp_dir.name] = zero
    return out


def main() -> int:
    setup_logging("INFO")
    log = get_logger("viz")
    p = argparse.ArgumentParser()
    p.add_argument("--runs_root", default="outputs")
    p.add_argument("--figures_root", default="figures")
    p.add_argument("--replot", default=None, help="If set, replot from a single JSON path.")
    p.add_argument("--out", default=None, help="Output SVG path (used with --replot).")
    args = p.parse_args()

    figs = Path(args.figures_root); figs.mkdir(exist_ok=True)
    if args.replot:
        target = Path(args.out or Path(args.replot).with_suffix(".svg"))
        for mod in (training_curves, roc_pr, confusion_matrix, tsne_umap, frequency_spectrum,
                    ablation_plot, param_efficiency, robustness_curves, calibration, arch,
                    recall_grid, prototype_geometry, real_manifold, spectrum_comparison,
                    cnd_disentanglement):
            try:
                mod.replot_from_json(args.replot, str(target))
                log.info(f"re-plotted via {mod.__name__} → {target}")
                return 0
            except Exception as e:
                log.debug(f"{mod.__name__} skipped: {e}")
        log.error("no replot module matched")
        return 1

    runs_root = Path(args.runs_root)
    summary_csv = figs / "runs_summary.csv"
    if not summary_csv.exists():
        log.warning("runs_summary.csv missing; run scripts/compile_results.py first")
    arch.draw_architecture(figs / "architecture.svg")
    arch.draw_pipeline(figs / "data_pipeline.svg")
    curves = training_curves.gather_curves(runs_root, ["E00_baseline", "E01_full"])
    if curves:
        training_curves.plot_training_curves(curves, figs / "training_curves.svg")
    runs = _runs_for_curves(runs_root)
    if runs:
        roc_pr.plot_rocs(runs, "indomain", figs / "roc_indomain.svg")
        roc_pr.plot_rocs(runs, "crossdomain", figs / "roc_crossdomain.svg")
        roc_pr.plot_prs(runs, "indomain", figs / "pr_indomain.svg")
        roc_pr.plot_prs(runs, "crossdomain", figs / "pr_crossdomain.svg")
    e01 = runs_root / "E01_full" / "0" / "predictions.npz"
    if e01.exists():
        confusion_matrix.plot_confusion(e01, "indomain", figs / "confusion_indomain.svg")
        confusion_matrix.plot_confusion(e01, "crossdomain", figs / "confusion_crossdomain.svg")
    e01_emb = runs_root / "E01_full" / "0" / "embeddings.npz"
    if e01_emb.exists():
        tsne_umap.plot_projection(e01_emb, figs / "tsne_embeddings.svg", method="tsne")
        tsne_umap.plot_projection(e01_emb, figs / "umap_embeddings.svg", method="umap")
    crops_root = Path("preprocessed/ffpp/face_crops")
    if crops_root.is_dir():
        samples: dict[str, list] = {"real": [], "Deepfakes": [], "Face2Face": [], "FaceSwap": [], "NeuralTextures": []}
        for cls in samples:
            d = crops_root / cls
            if not d.is_dir():
                continue
            for f in list(d.rglob("*.jpg"))[:200]:
                im = cv2.imread(str(f), cv2.IMREAD_COLOR)
                if im is None:
                    continue
                samples[cls].append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        samples = {k: v for k, v in samples.items() if v}
        if samples:
            frequency_spectrum.plot_radial_spectra(samples, figs / "frequency_spectrum.svg")

    if summary_csv.exists():
        ablation_plot.plot_per_manipulation_auc(summary_csv, figs / "per_manipulation_auc.svg")
        ablation_plot.plot_ablation_drops(summary_csv, figs / "ablation_bar.svg")

    baselines = figs / "external_baselines.json"
    if summary_csv.exists() and baselines.exists():
        df = pd.read_csv(summary_csv)
        e01_df = df[df["experiment_id"] == "E01_full"]
        if not e01_df.empty:
            our = {"WISER": (int(e01_df["params"].iloc[0]), float(e01_df["crossdomain_frame_auc"].mean()))}
            param_efficiency.plot_param_vs_auc(baselines, our, figs / "param_efficiency.svg")

    if (figs / "robustness_indomain.json").exists():
        robustness_curves.plot_robustness(
            figs / "robustness_indomain.json", figs / "robustness_curves.svg",
            crossdomain_json=figs / "robustness_crossdomain.json"
            if (figs / "robustness_crossdomain.json").exists() else None,
        )

    if e01.exists():
        calibration.plot_reliability(e01, "indomain", figs / "calibration_indomain.svg")
        calibration.plot_reliability(e01, "crossdomain", figs / "calibration_crossdomain.svg")

    if summary_csv.exists():
        try:
            recall_grid.plot_real_fake_recall(summary_csv, figs / "real_fake_recall_grid.svg")
        except Exception as e:
            log.debug(f"recall_grid skipped: {e}")

    e14 = runs_root / "E14_wiser_rf_full" / "0" / "predictions.npz"
    if e01.exists() and e14.exists():
        try:
            calibration.plot_reliability_grid(e01, e14, figs / "calibration_comparison.svg",
                                              side="crossdomain")
        except Exception as e:
            log.debug(f"calibration_comparison skipped: {e}")

    e01_ckpt = runs_root / "E01_full" / "0" / "ckpt" / "best.pt"
    e14_ckpt = runs_root / "E14_wiser_rf_full" / "0" / "ckpt" / "best.pt"
    if e01_ckpt.exists() or e14_ckpt.exists():
        ckpts = {}
        if e01_ckpt.exists():
            ckpts["E01 (MPHC)"] = e01_ckpt
        if e14_ckpt.exists():
            ckpts["E14 (RP-MPHC)"] = e14_ckpt
        try:
            prototype_geometry.plot_prototypes(ckpts, figs / "prototype_geometry.svg")
        except Exception as e:
            log.debug(f"prototype_geometry skipped: {e}")

    e14_emb = runs_root / "E14_wiser_rf_full" / "0" / "embeddings.npz"
    if e01_emb.exists() and e14_emb.exists():
        try:
            real_manifold.plot_real_manifold(e01_emb, e14_emb,
                                             figs / "real_manifold_umap.svg",
                                             titles=("E01", "E14"))
        except Exception as e:
            log.debug(f"real_manifold skipped: {e}")

    crops_root = Path("preprocessed/ffpp/face_crops/real")
    cdf_crops = Path("preprocessed/celebdfpp/face_crops")
    ffpp_fake_crops = Path("preprocessed/ffpp/face_crops")
    if crops_root.is_dir():
        import cv2  # type: ignore[import-not-found]
        import numpy as np

        real_paths = list(crops_root.rglob("*.jpg"))[:200]
        real_imgs = []
        for p in real_paths:
            im = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if im is None:
                continue
            real_imgs.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        cdf_imgs = []
        if cdf_crops.is_dir():
            cdf_paths = [p for p in cdf_crops.rglob("*.jpg") if "real" not in p.parts][:200]
            for p in cdf_paths:
                im = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if im is None:
                    continue
                cdf_imgs.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        if real_imgs:
            try:
                spectrum_comparison.plot_spectrum_comparison(
                    real_imgs, cdf_imgs, figs / "awb_sbi_spectrum_comparison.svg",
                )
            except Exception as e:
                log.debug(f"spectrum_comparison skipped: {e}")
        ffpp_fake_imgs: list = []
        if ffpp_fake_crops.is_dir():
            for cls in ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"):
                d = ffpp_fake_crops / cls
                if not d.is_dir():
                    continue
                for p in list(d.rglob("*.jpg"))[:50]:
                    im = cv2.imread(str(p), cv2.IMREAD_COLOR)
                    if im is None:
                        continue
                    ffpp_fake_imgs.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        if real_imgs and ffpp_fake_imgs:
            try:
                spectrum_comparison.plot_awb_v2_spectrum_comparison(
                    real_imgs, ffpp_fake_imgs,
                    figs / "awb_sbi_v2_spectrum_comparison.svg",
                )
            except Exception as e:
                log.debug(f"awb_v2 spectrum_comparison skipped: {e}")

    log.info(f"figures written under {figs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())