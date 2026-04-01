"""
Ground-Truth Evaluation — Excavator Activity Recognition
=========================================================
Reads ground-truth segments directly from Tasks.xlsx (Day2 & Day3 sheets),
compares against Part-10's frame_predictions.csv, and produces:

  evaluation/
    frame_comparison.csv        per-frame GT vs predicted
    segment_report.csv          per-segment accuracy + breakdown
    confusion_matrix.png        row-normalised heatmap
    timeline_Day2.png           dual-track timeline for Day 2
    timeline_Day3.png           dual-track timeline for Day 3

Run:  python evaluate_gt.py
"""

import re, csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker  as ticker
from pathlib import Path
from collections import defaultdict

# ──────────────────────────────────────────────
# PATHS  (edit these two lines)
# ──────────────────────────────────────────────
TASKS_XLSX      = r"C:\Users\shubh\Desktop\DELETE AFTER USE\Tasks.xlsx"
PREDICTIONS_CSV = r"C:\Users\shubh\Desktop\DELETE AFTER USE\FINAL Codes and results from Remote\frame_predictions.csv"
OUTPUT_DIR      = Path(PREDICTIONS_CSV).parent / "evaluation"

TARGET_FPS  = 25
ALL_CLASSES = ["digging", "idling", "loading", "swinging", "travelling"]

ACTIVITY_COLORS = {
    "digging"   : "#E74C3C",
    "idling"    : "#95A5A6",
    "loading"   : "#2ECC71",
    "swinging"  : "#3498DB",
    "travelling": "#F39C12",
}

# ──────────────────────────────────────────────
# 1.  Robust time parser
# ──────────────────────────────────────────────

def parse_time_range(raw: str):
    """'MM:SS - MM:SS'  (any spacing / dash variant)  →  (start_s, end_s) as floats."""
    parts = re.split(r'\s*-\s*', raw.strip(), maxsplit=1)
    times = []
    for p in parts:
        p = p.strip().replace(" ", "")          # kill spaces inside "136: 11" etc.
        m, s = p.split(":")
        times.append(int(m) * 60 + int(s))
    return float(times[0]), float(times[1])


# ──────────────────────────────────────────────
# 2.  Load ground truth from xlsx
# ──────────────────────────────────────────────

def load_gt_from_xlsx(xlsx_path):
    """Returns dict  { sheet_name: [(start_s, end_s, label), ...] }"""
    xls   = pd.ExcelFile(xlsx_path)
    result = {}
    for sheet in xls.sheet_names:
        df = pd.read_excel(xlsx_path, sheet_name=sheet, header=None)
        # Row 0 is the header row ("Time", "Activity", …) — skip it
        segments = []
        for idx in range(1, len(df)):
            time_raw  = str(df.iloc[idx, 0]).strip()
            label_raw = str(df.iloc[idx, 1]).strip().lower()
            if time_raw in ("nan", "") or label_raw in ("nan", ""):
                continue
            try:
                start_s, end_s = parse_time_range(time_raw)
            except Exception:
                print(f"  [WARN] Could not parse time '{time_raw}' in {sheet} row {idx+1} — skipped")
                continue
            if label_raw not in ALL_CLASSES:
                print(f"  [WARN] Unknown label '{label_raw}' in {sheet} row {idx+1} — skipped")
                continue
            segments.append((start_s, end_s, label_raw))
        result[sheet] = segments
        print(f"  {sheet}: {len(segments)} raw segments loaded")
    return result


# ──────────────────────────────────────────────
# 3.  Merge consecutive same-label segments
# ──────────────────────────────────────────────

def merge_consecutive(segments):
    if not segments:
        return []
    merged = [list(segments[0])]
    for start, end, label in segments[1:]:
        if label == merged[-1][2] and abs(start - merged[-1][1]) < 0.5:
            merged[-1][1] = end
        else:
            merged.append([start, end, label])
    return [tuple(s) for s in merged]


# ──────────────────────────────────────────────
# 4.  Load predictions CSV
# ──────────────────────────────────────────────

def load_predictions(csv_path):
    preds = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["Frame"])
            preds[idx] = {
                "time_s"    : float(row["Time_s"]),
                "activity"  : row["Activity"].strip().lower(),
                "confidence": float(row["Confidence"])
            }
    return preds


# ──────────────────────────────────────────────
# 5.  Build GT frame map  &  run comparison
# ──────────────────────────────────────────────

def build_gt_frame_map(segments, fps):
    gt_map = {}
    for start_s, end_s, label in segments:
        sf = int(round(start_s * fps))
        ef = int(round(end_s   * fps))
        for f in range(sf, ef):
            gt_map[f] = label
    return gt_map


def compare_frames(gt_map, predictions):
    rows = []
    for frame_idx in sorted(gt_map.keys()):
        if frame_idx not in predictions:
            continue
        gt_lab   = gt_map[frame_idx]
        pred_lab = predictions[frame_idx]["activity"]
        conf     = predictions[frame_idx]["confidence"]
        rows.append((frame_idx, gt_lab, pred_lab, conf, gt_lab == pred_lab))
    return rows


# ──────────────────────────────────────────────
# 6.  Per-segment report
# ──────────────────────────────────────────────

def segment_report(segments, predictions, fps):
    rows = []
    for start_s, end_s, gt_label in segments:
        sf = int(round(start_s * fps))
        ef = int(round(end_s   * fps))
        pred_counts = defaultdict(int)
        total = correct = 0
        for f in range(sf, ef):
            if f not in predictions:
                continue
            total += 1
            pred_lab = predictions[f]["activity"]
            pred_counts[pred_lab] += 1
            if pred_lab == gt_label:
                correct += 1
        accuracy = correct / total if total else 0.0
        dominant = max(pred_counts, key=pred_counts.get) if pred_counts else "—"
        rows.append({
            "start"         : start_s,
            "end"           : end_s,
            "gt_label"      : gt_label,
            "dominant_pred" : dominant,
            "accuracy"      : accuracy,
            "correct"       : correct,
            "total_frames"  : total,
            "pred_breakdown": dict(pred_counts),
        })
    return rows


# ──────────────────────────────────────────────
# 7.  Confusion matrix
# ──────────────────────────────────────────────

def compute_confusion(comparison_rows, classes):
    n   = len(classes)
    idx = {c: i for i, c in enumerate(classes)}
    cm  = np.zeros((n, n), dtype=int)
    for _, gt, pred, _, _ in comparison_rows:
        if gt in idx and pred in idx:
            cm[idx[gt]][idx[pred]] += 1
    return cm


def plot_confusion(cm, classes, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm  = cm / row_sums

    im = ax.imshow(cm_norm, cmap="YlOrRd", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]*100:.0f}%)",
                    ha="center", va="center", fontsize=9, color=color)

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=10)
    ax.set_yticklabels(classes, fontsize=10)
    ax.set_ylabel("Ground Truth",  fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted",     fontsize=12, fontweight="bold")
    ax.set_title("Confusion Matrix (row-normalised)", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Fraction predicted as column class")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────
# 8.  Dual-timeline plot  (per day)
# ──────────────────────────────────────────────

def rle_predictions(predictions, start_frame, end_frame):
    """Run-length encode predictions in a frame range."""
    segs = []
    cur_label = None
    seg_start = None
    for f in range(start_frame, end_frame):
        lab = predictions[f]["activity"] if f in predictions else None
        if lab != cur_label:
            if cur_label is not None:
                segs.append((seg_start, f, cur_label))
            cur_label = lab
            seg_start = f
    if cur_label is not None:
        segs.append((seg_start, end_frame, cur_label))
    return segs


def plot_timeline(segments, predictions, fps, save_path, title):
    gt_start = segments[0][0]
    gt_end   = segments[-1][1]
    margin   = 5
    t_min    = max(0, gt_start - margin)
    t_max    = gt_end + margin

    sf = int(round(t_min * fps))
    ef = int(round(t_max * fps))
    pred_segments = [(s/fps, e/fps, lab) for s, e, lab in rle_predictions(predictions, sf, ef) if lab]

    fig, ax = plt.subplots(figsize=(20, 5))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")

    track_gt   = 1.4
    track_pred = 0.4
    bar_h      = 0.7

    # --- ground truth ---
    for s, e, label in segments:
        color = ACTIVITY_COLORS.get(label, "#ccc")
        ax.barh(track_gt, e - s, left=s, height=bar_h,
                color=color, edgecolor="white", linewidth=1.2)
        if (e - s) > 1.5:
            ax.text((s + e) / 2, track_gt, label,
                    ha="center", va="center", fontsize=7, fontweight="bold", color="white")

    # --- predictions ---
    for s, e, label in pred_segments:
        color = ACTIVITY_COLORS.get(label, "#ccc")
        ax.barh(track_pred, e - s, left=s, height=bar_h,
                color=color, edgecolor="white", linewidth=1.2)
        if (e - s) > 1.5:
            ax.text((s + e) / 2, track_pred, label,
                    ha="center", va="center", fontsize=7, fontweight="bold", color="white")

    # --- GT boundary lines ---
    for s, e, _ in segments:
        ax.axvline(s, color="#aaa", linewidth=0.6, linestyle="--", zorder=0)
        ax.axvline(e, color="#aaa", linewidth=0.6, linestyle="--", zorder=0)

    ax.set_xlim(t_min, t_max)
    ax.set_ylim(0, 2)
    ax.set_yticks([track_pred, track_gt])
    ax.set_yticklabels(["Predicted", "Ground Truth"], fontsize=11, fontweight="bold")

    def fmt_x(val, pos):
        m = int(val) // 60
        s = val - m * 60
        return f"{m:02d}:{s:04.1f}"
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_x))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    ax.set_xlabel("Time", fontsize=11)

    handles = [mpatches.Patch(color=ACTIVITY_COLORS[c], label=c.capitalize()) for c in ALL_CLASSES]
    ax.legend(handles=handles, loc="upper right", frameon=True, fontsize=9)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────
# 9.  Per-class precision / recall / F1
# ──────────────────────────────────────────────

def class_metrics(comparison_rows, classes):
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    for _, gt, pred, _, _ in comparison_rows:
        if gt == pred:
            tp[gt] += 1
        else:
            fn[gt] += 1
            fp[pred] += 1

    print(f"\n  {'Class':<14} {'TP':>5} {'FP':>5} {'FN':>5} "
          f"{'Prec':>7} {'Recall':>7} {'F1':>7} {'Support':>8}")
    print("  " + "-" * 68)
    f1s = []
    for c in classes:
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        rec  = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        f1   = 2*prec*rec / (prec+rec)  if (prec+rec)       else 0.0
        f1s.append(f1)
        print(f"  {c:<14} {tp[c]:>5} {fp[c]:>5} {fn[c]:>5} "
              f"{prec*100:>6.1f}% {rec*100:>6.1f}% {f1*100:>6.1f}% {tp[c]+fn[c]:>8}")

    present = [i for i, c in enumerate(classes) if (tp[c] + fn[c]) > 0]
    if present:
        print("  " + "-" * 68)
        print(f"  {'Macro avg':<14} {'':>5} {'':>5} {'':>5} "
              f"{'':>7} {'':>7} {np.mean([f1s[i] for i in present])*100:>6.1f}%")

    total   = len(comparison_rows)
    correct = sum(1 for *_, m in comparison_rows if m)
    print(f"\n  Overall frame-level accuracy: {correct}/{total}  ({correct/total*100:.1f}%)")


# ──────────────────────────────────────────────
# 10.  Main
# ──────────────────────────────────────────────

def fmt_mm_ss(s):
    m = int(s) // 60
    sec = s - m * 60
    return f"{m:02d}:{sec:05.2f}"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 65)
    print("  GROUND-TRUTH EVALUATION")
    print("=" * 65)

    # --- load GT ---
    print("\n[1] Loading ground truth from Tasks.xlsx …")
    gt_raw = load_gt_from_xlsx(TASKS_XLSX)

    # --- load predictions ---
    print("\n[2] Loading predictions …")
    predictions = load_predictions(PREDICTIONS_CSV)
    print(f"  {len(predictions)} predicted frames")

    # ── determine which sheet covers the prediction range ──
    # predictions are indexed from frame 0; find their max time
    if predictions:
        max_pred_time = max(p["time_s"] for p in predictions.values())
        print(f"  Prediction range: 0.0 s  –  {max_pred_time:.1f} s  "
              f"({max_pred_time/60:.2f} min)")

    # ── process each sheet ──
    all_comparison = []          # accumulate across sheets for global metrics

    for sheet_name, raw_segs in gt_raw.items():
        print(f"\n{'─'*65}")
        print(f"  SHEET: {sheet_name}")
        print(f"{'─'*65}")

        # merge
        merged = merge_consecutive(raw_segs)
        print(f"  {len(raw_segs)} raw  →  {len(merged)} after merging")

        # check overlap with prediction range
        sheet_start = merged[0][0]
        sheet_end   = merged[-1][1]
        print(f"  GT range: {fmt_mm_ss(sheet_start)}  –  {fmt_mm_ss(sheet_end)}")

        if predictions and sheet_start > max_pred_time:
            print(f"  ⚠  Entire sheet is BEYOND prediction range — skipped")
            continue

        # clip merged segments to prediction range
        clipped = []
        for s, e, lab in merged:
            if predictions and e > max_pred_time:
                e = max_pred_time
            if s < e:
                clipped.append((s, e, lab))
        if not clipped:
            print(f"  ⚠  No segments overlap predictions — skipped")
            continue

        # frame map
        gt_map     = build_gt_frame_map(clipped, TARGET_FPS)
        comparison = compare_frames(gt_map, predictions)
        all_comparison.extend(comparison)

        if not comparison:
            print(f"  ⚠  Zero matching frames — check time alignment")
            continue

        # ── frame CSV ──
        frame_csv = OUTPUT_DIR / f"frame_comparison_{sheet_name}.csv"
        with open(frame_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Frame", "Time_s", "GT_Label", "Pred_Label", "Confidence", "Match"])
            for fi, gt, pred, conf, match in comparison:
                w.writerow([fi, f"{fi/TARGET_FPS:.3f}", gt, pred, f"{conf:.4f}", match])
        print(f"  Saved: {frame_csv}")

        # ── per-class metrics ──
        print(f"\n  PER-CLASS METRICS  ({sheet_name})")
        class_metrics(comparison, ALL_CLASSES)

        # ── segment report ──
        print(f"\n  PER-SEGMENT REPORT  ({sheet_name})")
        seg_rows = segment_report(clipped, predictions, TARGET_FPS)

        seg_csv = OUTPUT_DIR / f"segment_report_{sheet_name}.csv"
        with open(seg_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Start", "End", "GT_Label", "Dominant_Pred", "Accuracy_%",
                        "Correct", "Total_Frames", "Pred_Breakdown"])
            for r in seg_rows:
                w.writerow([fmt_mm_ss(r["start"]), fmt_mm_ss(r["end"]),
                            r["gt_label"], r["dominant_pred"],
                            f"{r['accuracy']*100:.1f}", r["correct"],
                            r["total_frames"], str(r["pred_breakdown"])])

        print(f"\n  {'Segment':<22} {'GT':<12} {'Pred':<12} {'Acc':>6} {'Fr':>5}  Breakdown")
        print("  " + "-" * 82)
        for r in seg_rows:
            bd = "  ".join(f"{k}:{v}" for k, v in
                           sorted(r["pred_breakdown"].items(), key=lambda x: -x[1]))
            mark = "✓" if r["accuracy"] > 0.7 else ("~" if r["accuracy"] > 0.4 else "✗")
            print(f"  {fmt_mm_ss(r['start'])+' – '+fmt_mm_ss(r['end']):<22} "
                  f"{r['gt_label']:<12} {r['dominant_pred']:<12} "
                  f"{r['accuracy']*100:>5.1f}% {r['total_frames']:>4}  {bd}  {mark}")
        print(f"  Saved: {seg_csv}")

        # ── confusion matrix (per sheet) ──
        cm = compute_confusion(comparison, ALL_CLASSES)
        plot_confusion(cm, ALL_CLASSES, OUTPUT_DIR / f"confusion_matrix_{sheet_name}.png")

        # ── timeline ──
        plot_timeline(clipped, predictions, TARGET_FPS,
                      OUTPUT_DIR / f"timeline_{sheet_name}.png",
                      f"Activity Timeline — Ground Truth vs Predicted  ({sheet_name})")

    # ── GLOBAL metrics across all sheets ──
    if all_comparison:
        print(f"\n{'='*65}")
        print("  GLOBAL METRICS  (all sheets combined)")
        print(f"{'='*65}")
        class_metrics(all_comparison, ALL_CLASSES)

        cm_all = compute_confusion(all_comparison, ALL_CLASSES)
        plot_confusion(cm_all, ALL_CLASSES, OUTPUT_DIR / "confusion_matrix_ALL.png")

    # ── summary ──
    total   = len(all_comparison)
    correct = sum(1 for *_, m in all_comparison if m)
    print(f"\n{'='*65}")
    print("  SUMMARY")
    print(f"{'='*65}")
    print(f"  Total evaluated frames : {total}")
    print(f"  Overall accuracy       : {correct}/{total}  ({correct/total*100:.1f}%)" if total else "  No frames evaluated")
    print(f"  Outputs in             : {OUTPUT_DIR}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()