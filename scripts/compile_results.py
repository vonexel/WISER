from __future__ import annotations

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from wiser.utils import save_csv, save_json
from wiser.utils.logging import get_logger, setup_logging

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

WISER_BASE_HEADER = [ "experiment_id", "seed", "params", "wallclock_seconds",
                      "indomain_frame_auc", "indomain_frame_ap", "indomain_frame_eer",
                      "indomain_frame_acc05", "indomain_frame_acc_opt", "indomain_frame_f1",
                      "indomain_frame_brier", "indomain_frame_ece",
                      "indomain_video_auc", "indomain_video_eer",
                      "indomain_DF_auc", "indomain_F2F_auc", "indomain_FS_auc",
                      "indomain_NT_auc", "indomain_FSh_auc",
                      "crossdomain_frame_auc", "crossdomain_frame_ap", "crossdomain_frame_eer",
                      "crossdomain_frame_acc05", "crossdomain_frame_acc_opt", "crossdomain_frame_f1",
                      "crossdomain_frame_brier", "crossdomain_frame_ece",
                      "crossdomain_video_auc", "crossdomain_video_eer",
                      "crossdomain_FS_auc", "crossdomain_FR_auc", "crossdomain_TF_auc"]

WISER_RF_EXTRA_HEADER = ["indomain_real_recall_uncal", "indomain_fake_recall_uncal", "indomain_balanced_acc_uncal",
                         "indomain_real_recall_cal", "indomain_fake_recall_cal", "indomain_balanced_acc_cal",
                         "crossdomain_real_recall_uncal", "crossdomain_fake_recall_uncal", "crossdomain_balanced_acc_uncal",
                         "crossdomain_real_recall_cal", "crossdomain_fake_recall_cal", "crossdomain_balanced_acc_cal",
                         "crossdomain_frame_auc_uncal", "crossdomain_frame_auc_cal",
                         "crossdomain_fpr_at_tpr95_cal", "crossdomain_tpr_at_fpr05_cal",
                         "ece_real_indomain_cal", "ece_fake_indomain_cal",
                         "ece_real_crossdomain_cal", "ece_fake_crossdomain_cal",
                         "calib_temperature", "calib_prior_bias", "calib_threshold"]

CSV_HEADER = WISER_BASE_HEADER + WISER_RF_EXTRA_HEADER

DEFAULT_BASELINES = {"MesoNet":        {"params": 27977,    "crossdomain_auc": 0.840, "source": "Afchar+2018"},
                     "Xception":       {"params": 22855952, "crossdomain_auc": 0.737, "source": "Rossler+2019"},
                     "EfficientNetB4": {"params": 17673823, "crossdomain_auc": 0.841, "source": "Rossler+2019"},
                     "F3-Net":         {"params": 24400000, "crossdomain_auc": 0.654, "source": "Qian+2020"},
                     "FDFL":           {"params": 23330000, "crossdomain_auc": 0.671, "source": "Li+2021"},
                     "SBI":            {"params": 23330000, "crossdomain_auc": 0.937, "source": "Shiohara+2022"},
                     "FSBI":           {"params": 23330000, "crossdomain_auc": 0.949, "source": "Sabri+2024"}}


def _g(d: dict, k: str) -> float:
    v = d.get(k) if isinstance(d, dict) else None
    return float(v) if v is not None else float("nan")


def _legacy_block(metrics: dict) -> tuple[dict, dict]:
    if metrics.get("schema_version") == "2.0":
        uncal = metrics.get("uncalibrated", {})
        return uncal.get("indomain", {}), uncal.get("crossdomain", {})
    return metrics.get("indomain", {}), metrics.get("crossdomain", {})


def _row(metrics: dict) -> dict:
    in_block, cd_block = _legacy_block(metrics)
    in_frame = in_block.get("frame", {})
    in_video = in_block.get("video", {})
    in_per = in_block.get("per_manipulation", {})
    cd_frame = cd_block.get("frame", {}) if isinstance(cd_block, dict) else {}
    cd_video = cd_block.get("video", {}) if isinstance(cd_block, dict) else {}
    cd_per = cd_block.get("per_category", {}) if isinstance(cd_block, dict) else {}

    base = {
        "experiment_id": metrics.get("experiment_id", "?"),
        "seed": int(metrics.get("seed", -1)),
        "params": int(metrics.get("params_total", 0)),
        "wallclock_seconds": float(metrics.get("wallclock_seconds", 0.0)),
        "indomain_frame_auc": _g(in_frame, "auc"),
        "indomain_frame_ap": _g(in_frame, "ap"),
        "indomain_frame_eer": _g(in_frame, "eer"),
        "indomain_frame_acc05": _g(in_frame, "acc_at_05"),
        "indomain_frame_acc_opt": _g(in_frame, "acc_at_optimal"),
        "indomain_frame_f1": _g(in_frame, "f1"),
        "indomain_frame_brier": _g(in_frame, "brier"),
        "indomain_frame_ece": _g(in_frame, "ece"),
        "indomain_video_auc": _g(in_video, "auc"),
        "indomain_video_eer": _g(in_video, "eer"),
        "indomain_DF_auc": _g(in_per.get("Deepfakes", {}), "auc"),
        "indomain_F2F_auc": _g(in_per.get("Face2Face", {}), "auc"),
        "indomain_FS_auc": _g(in_per.get("FaceSwap", {}), "auc"),
        "indomain_NT_auc": _g(in_per.get("NeuralTextures", {}), "auc"),
        "indomain_FSh_auc": _g(in_per.get("FaceShifter", {}), "auc"),
        "crossdomain_frame_auc": _g(cd_frame, "auc"),
        "crossdomain_frame_ap": _g(cd_frame, "ap"),
        "crossdomain_frame_eer": _g(cd_frame, "eer"),
        "crossdomain_frame_acc05": _g(cd_frame, "acc_at_05"),
        "crossdomain_frame_acc_opt": _g(cd_frame, "acc_at_optimal"),
        "crossdomain_frame_f1": _g(cd_frame, "f1"),
        "crossdomain_frame_brier": _g(cd_frame, "brier"),
        "crossdomain_frame_ece": _g(cd_frame, "ece"),
        "crossdomain_video_auc": _g(cd_video, "auc"),
        "crossdomain_video_eer": _g(cd_video, "eer"),
        "crossdomain_FS_auc": _g(cd_per.get("FaceSwap", {}), "auc"),
        "crossdomain_FR_auc": _g(cd_per.get("FaceReenactment", {}), "auc"),
        "crossdomain_TF_auc": _g(cd_per.get("TalkingFace", {}), "auc"),
    }

    extra = _wiser_rf_columns(metrics, in_frame, cd_frame, in_per, cd_per, in_block, cd_block)
    base.update(extra)
    return base


def _wiser_rf_columns(metrics: dict, in_frame: dict, cd_frame: dict, in_per: dict, cd_per: dict, in_block: dict, cd_block: dict) -> dict:
    schema = metrics.get("schema_version", "1.0")
    out: dict = {k: float("nan") for k in WISER_RF_EXTRA_HEADER}
    if schema == "2.0":
        uncal = metrics.get("uncalibrated", {})
        cal = metrics.get("calibrated", {})
        u_in_frame = uncal.get("indomain", {}).get("frame", {})
        u_cd_frame = uncal.get("crossdomain", {}).get("frame", {})
        c_in_frame = cal.get("indomain", {}).get("frame", {})
        c_cd_frame = cal.get("crossdomain", {}).get("frame", {})
    else:
        u_in_frame, c_in_frame = in_frame, in_frame
        u_cd_frame, c_cd_frame = cd_frame, cd_frame

    out["indomain_real_recall_uncal"] = _g(u_in_frame, "real_recall")
    out["indomain_fake_recall_uncal"] = _g(u_in_frame, "fake_recall")
    out["indomain_balanced_acc_uncal"] = _g(u_in_frame, "balanced_acc")
    out["indomain_real_recall_cal"] = _g(c_in_frame, "real_recall")
    out["indomain_fake_recall_cal"] = _g(c_in_frame, "fake_recall")
    out["indomain_balanced_acc_cal"] = _g(c_in_frame, "balanced_acc")
    out["crossdomain_real_recall_uncal"] = _g(u_cd_frame, "real_recall")
    out["crossdomain_fake_recall_uncal"] = _g(u_cd_frame, "fake_recall")
    out["crossdomain_balanced_acc_uncal"] = _g(u_cd_frame, "balanced_acc")
    out["crossdomain_real_recall_cal"] = _g(c_cd_frame, "real_recall")
    out["crossdomain_fake_recall_cal"] = _g(c_cd_frame, "fake_recall")
    out["crossdomain_balanced_acc_cal"] = _g(c_cd_frame, "balanced_acc")
    out["crossdomain_frame_auc_uncal"] = _g(u_cd_frame, "auc")
    out["crossdomain_frame_auc_cal"] = _g(c_cd_frame, "auc")
    out["crossdomain_fpr_at_tpr95_cal"] = _g(c_cd_frame, "fpr_at_tpr95")
    out["crossdomain_tpr_at_fpr05_cal"] = _g(c_cd_frame, "tpr_at_fpr05")
    out["ece_real_indomain_cal"] = _g(c_in_frame, "ece_real")
    out["ece_fake_indomain_cal"] = _g(c_in_frame, "ece_fake")
    out["ece_real_crossdomain_cal"] = _g(c_cd_frame, "ece_real")
    out["ece_fake_crossdomain_cal"] = _g(c_cd_frame, "ece_fake")

    cal_block = metrics.get("calibration", {}) if isinstance(metrics, dict) else {}
    out["calib_temperature"] = _g(cal_block, "temperature")
    out["calib_prior_bias"] = _g(cal_block, "prior_bias")
    out["calib_threshold"] = _g(cal_block, "threshold")
    return out


def _booktabs_main(df: pd.DataFrame) -> str:
    panel = df[df["experiment_id"].str.match(r"^E0[0-9]_")]
    agg = panel.groupby("experiment_id").agg(
        params=("params", "mean"),
        in_auc_mean=("indomain_frame_auc", "mean"),
        in_auc_std=("indomain_frame_auc", "std"),
        cd_auc_mean=("crossdomain_frame_auc", "mean"),
        cd_auc_std=("crossdomain_frame_auc", "std"),
    ).reset_index()
    rows = []
    for _, r in agg.iterrows():
        rows.append(
            f"{r['experiment_id']} & {int(r['params']):,} & "
            f"{r['in_auc_mean']:.3f}$\\pm${r['in_auc_std']:.3f} & "
            f"{r['cd_auc_mean']:.3f}$\\pm${r['cd_auc_std']:.3f}\\\\")
    body = "\n".join(rows)
    return (
        "\\begin{tabular}{lrrr}\n\\toprule\n"
        "Variant & Params & In-AUC & Cross-AUC \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n")


def _booktabs_wiser_rf(df: pd.DataFrame) -> str:
    panel_ids = ["E01_full"] + [f"E{i:02d}" for i in range(10, 19)]
    panel = df[df["experiment_id"].apply(lambda x: any(x.startswith(p) for p in panel_ids))]
    agg = panel.groupby("experiment_id").agg(
        params=("params", "mean"),
        cd_auc_cal_mean=("crossdomain_frame_auc_cal", "mean"),
        cd_auc_cal_std=("crossdomain_frame_auc_cal", "std"),
        cd_real_cal_mean=("crossdomain_real_recall_cal", "mean"),
        cd_real_cal_std=("crossdomain_real_recall_cal", "std"),
        cd_fake_cal_mean=("crossdomain_fake_recall_cal", "mean"),
        cd_fake_cal_std=("crossdomain_fake_recall_cal", "std"),
        cd_bal_cal_mean=("crossdomain_balanced_acc_cal", "mean"),
        cd_bal_cal_std=("crossdomain_balanced_acc_cal", "std"),
    ).reset_index()
    rows = []
    for _, r in agg.iterrows():
        rows.append(
            f"{r['experiment_id']} & {int(r['params']):,} & "
            f"{r['cd_bal_cal_mean']:.3f}$\\pm${r['cd_bal_cal_std']:.3f} & "
            f"{r['cd_real_cal_mean']:.3f}$\\pm${r['cd_real_cal_std']:.3f} & "
            f"{r['cd_fake_cal_mean']:.3f}$\\pm${r['cd_fake_cal_std']:.3f} & "
            f"{r['cd_auc_cal_mean']:.3f}$\\pm${r['cd_auc_cal_std']:.3f}\\\\"
        )
    body = "\n".join(rows) if rows else "% no rows"
    return (
        "\\begin{tabular}{lrrrrr}\n\\toprule\n"
        "Variant & Params & Bal-Acc (cal) & Real-Recall (cal) & Fake-Recall (cal) & Cross-AUC (cal)\\\\\n"
        "\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n"
    )


def _booktabs_main_comparison(df: pd.DataFrame) -> str:
    sub = df[df["experiment_id"].isin(["E01_full", "E04_wiser_rf_full"])]
    if sub.empty:
        return "% no rows for main comparison\n"
    agg = sub.groupby("experiment_id").agg(
        in_auc=("indomain_frame_auc", "mean"),
        cd_auc=("crossdomain_frame_auc_cal", "mean"),
        cd_bal=("crossdomain_balanced_acc_cal", "mean"),
        cd_real=("crossdomain_real_recall_cal", "mean"),
        cd_fake=("crossdomain_fake_recall_cal", "mean"),
        ece_real_cd=("ece_real_crossdomain_cal", "mean"),
    ).reset_index()
    rows = []
    for _, r in agg.iterrows():
        rows.append(
            f"{r['experiment_id']} & {r['in_auc']:.3f} & {r['cd_auc']:.3f} & "
            f"{r['cd_bal']:.3f} & {r['cd_real']:.3f} & {r['cd_fake']:.3f} & "
            f"{r['ece_real_cd']:.3f}\\\\"
        )
    return (
        "\\begin{tabular}{lrrrrrr}\n\\toprule\n"
        "Method & In-AUC & Cross-AUC & Cross-BalAcc & Real-Recall & Fake-Recall & ECE-real \\\\\n"
        "\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )


def _booktabs_calibration_lift(df: pd.DataFrame) -> str:
    agg = df.groupby("experiment_id").agg(
        bal_uncal=("crossdomain_balanced_acc_uncal", "mean"),
        bal_cal=("crossdomain_balanced_acc_cal", "mean"),
        real_uncal=("crossdomain_real_recall_uncal", "mean"),
        real_cal=("crossdomain_real_recall_cal", "mean")).reset_index()
    rows = []
    for _, r in agg.iterrows():
        lift = float(r["bal_cal"]) - float(r["bal_uncal"])
        rows.append(
            f"{r['experiment_id']} & {r['bal_uncal']:.3f} & {r['bal_cal']:.3f} & "
            f"{lift:+.3f} & {r['real_uncal']:.3f} & {r['real_cal']:.3f}\\\\"
        )
    return (
        "\\begin{tabular}{lrrrrr}\n\\toprule\n"
        "Cell & BalAcc (uncal) & BalAcc (cal) & $\\Delta$ & Real-Rec (uncal) & Real-Rec (cal) \\\\\n"
        "\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n")


def _significance(df: pd.DataFrame, *, reference: str, metric: str) -> dict:
    if reference not in df["experiment_id"].unique():
        return {}
    from scipy import stats

    ref = df[df["experiment_id"] == reference].sort_values("seed")[metric].values
    out: dict = {"metric": metric, "reference": reference, "tests": []}
    for exp in sorted(df["experiment_id"].unique()):
        if exp == reference:
            continue
        sub = df[df["experiment_id"] == exp].sort_values("seed")[metric].values
        if len(sub) != len(ref):
            continue
        delta = float(np.nanmean(ref) - np.nanmean(sub))
        valid = np.isfinite(ref) & np.isfinite(sub)
        if valid.sum() < 2:
            t, p = float("nan"), float("nan")
        else:
            t_stat = stats.ttest_rel(ref[valid], sub[valid])
            t, p = float(t_stat.statistic), float(t_stat.pvalue)
        out["tests"].append({"experiment": exp, "delta": delta, "t": t, "p_value": p})
    return out


def main() -> int:
    setup_logging("INFO")
    log = get_logger("compile")
    runs_root = Path("outputs")
    figures_root = Path("figures")
    figures_root.mkdir(exist_ok=True)

    baselines_path = figures_root / "external_baselines.json"
    if not baselines_path.exists():
        save_json(baselines_path, DEFAULT_BASELINES)

    rows = []
    for metrics_path in sorted(runs_root.glob("*/*/metrics.json")):
        with open(metrics_path) as f:
            metrics = json.load(f)
        metrics.setdefault("experiment_id", metrics_path.parent.parent.name)
        metrics.setdefault("seed", int(metrics_path.parent.name))
        rows.append(_row(metrics))

    if not rows:
        log.warning("no metrics.json files found under outputs/")
        save_csv(figures_root / "runs_summary.csv", CSV_HEADER, [])
        return 0

    df = pd.DataFrame(rows, columns=CSV_HEADER)
    df.to_csv(figures_root / "runs_summary.csv", index=False)
    log.info(f"wrote {figures_root / 'runs_summary.csv'} ({len(df)} rows)")

    (figures_root / "ablation_table.tex").write_text(_booktabs_main(df), encoding="utf-8")
    log.info(f"wrote {figures_root / 'ablation_table.tex'}")

    (figures_root / "wiser_rf_ablation_table.tex").write_text(_booktabs_wiser_rf(df), encoding="utf-8")
    log.info(f"wrote {figures_root / 'wiser_rf_ablation_table.tex'}")

    (figures_root / "main_comparison_table.tex").write_text(_booktabs_main_comparison(df), encoding="utf-8")
    log.info(f"wrote {figures_root / 'main_comparison_table.tex'}")

    (figures_root / "calibration_lift_table.tex").write_text(_booktabs_calibration_lift(df), encoding="utf-8")
    log.info(f"wrote {figures_root / 'calibration_lift_table.tex'}")

    sig = _significance(df, reference="E01_full", metric="crossdomain_frame_auc")
    if sig:
        save_json(figures_root / "significance.json", sig)
    sig_rf = _significance(df, reference="E04_wiser_rf_full", metric="crossdomain_balanced_acc_cal")
    if sig_rf:
        save_json(figures_root / "wiser_rf_significance.json", sig_rf)
    return 0


if __name__ == "__main__":
    sys.exit(main())