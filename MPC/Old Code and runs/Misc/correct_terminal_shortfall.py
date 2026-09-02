#!/usr/bin/env python3
# =============================================================================
# correct_terminal_shortfall.py
# -----------------------------------------------------------------------------
# ONE-TIME POST-PROCESSING FIX for the terminal-shortfall "realized vs required"
# change -- corrects already-generated Comparison_A0_A1_A2 sweep outputs
# WITHOUT re-running the optimization.
#
# WHY THIS IS SAFE AS POST-PROCESSING:
#   - Terminal_SOE_Shortfall_kWh is UNCHANGED by the fix -- it only depends on
#     the actual end-of-day battery levels the MILP already produced.
#   - avg_work_power_kW's REQUIRED-based ingredients (p_digging,
#     p_loading_swinging, hours_digging, hours_loading_swinging, rho_miss) are
#     plain INPUT constants in parameters.csv/place.csv -- not something the
#     MILP computes, so nothing needs to be re-solved.
#   - Total_Cost_USD is a flat additive sum of cost components (confirmed in
#     8_ComparisonOutput.jl's cost_components: energy + carbon + ncd + opd +
#     missed + travel + shortfall), so correcting just the shortfall term and
#     re-summing is exact.
#
# AUDITED, FILE BY FILE, AGAINST THE ACTUAL JULIA CODE (8_ComparisonOutput.jl)
# BEFORE WRITING THIS SCRIPT -- see the chat writeup for the full per-file
# table. Summary of what touches vs. doesn't touch cost:
#
#   UNCHANGED, copied through byte-identical (no cost numbers involved, or a
#   cost number that's untouched by this specific fix):
#     01-06, 07_approach_timeline_comparison.png, 09-13 (all .csv/.png),
#     run_log.txt, and the TOP-LEVEL mode_sweep_kpi_summary.csv (its
#     total_cost_USD column is ENERGY-ONLY cost -- sum(grid_kW .* price) --
#     confirmed in 4_MPCLoop.jl, and was NEVER affected by this bug; rebuilding
#     it from the aggregate Total_Cost_USD would silently change what the
#     column means, so it is deliberately left untouched here)
#     11_diagnostic_capacity_summary.csv DOES contain a $ figure
#     (Missed_Work_Penalty_USD = rho_miss * missed) but that is a different,
#     unaffected mechanism (live physical-floor missed work, not terminal
#     shortfall) -- copied through untouched too.
#
#   CORRECTED:
#     08_cost_kpi_metrics.csv    -- Terminal_Shortfall_Penalty_USD & Total_Cost_USD rows
#     08_kpi_metrics_summary.png -- regenerated bar chart (Shortfall & TOTAL bars)
#     *_vs_*.html                -- regenerated table (two stale rows)
#
#   DELIBERATELY SKIPPED (per instruction): 07_mcs_optimization_summary.png
#   (one stale text line inside an 8-panel composite; not worth rebuilding the
#   whole figure for one line -- read the corrected .html/.csv instead).
#
# USAGE:
#   python correct_terminal_shortfall.py
#   (edit INPUT_DIR / OUTPUT_DIR below first, or pass them as arguments:)
#   python correct_terminal_shortfall.py "C:\...\Input" "C:\...\Output"
# =============================================================================

import sys
import csv
import shutil
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

# ---- EDIT THESE TWO PATHS (or pass them as command-line arguments) ----
INPUT_DIR  = r"C:\Users\shubh\Desktop\MPC\Comparison_A0_A1_A2\Input"
OUTPUT_DIR = r"C:\Users\shubh\Desktop\MPC\Comparison_A0_A1_A2\Output"
# Corrected files are written HERE, alongside Output/. The original Output/
# folder is opened read-only and never modified.
OUTPUT_CORRECTED_DIR = None  # None -> auto: same parent, "Output_corrected"

# Matches the Approach struct colors in 7_Comparison_main_RecedingOnlyVersion.jl
APPROACH_COLORS = {"A0": "#666666", "A1R": "firebrick", "A2R": "darkorange"}

# Exact row order/labels from write_kpi_html in 8_ComparisonOutput.jl, mapped
# to the corresponding row name in 08_cost_kpi_metrics.csv.
HTML_ROWS = [
    ("Grid energy (kWh)",                 "Total_Grid_Energy_kWh"),
    ("Energy cost (USD)",                 "Total_Energy_Cost_USD"),
    ("CO2 emissions (kg)",                "Total_CO2_Emissions_kg"),
    ("CO2 cost (USD)",                    "Total_CO2_Cost_USD"),
    ("NCD peak (kW)",                     "NCD_Peak_kW"),
    ("NCD charge (USD)",                  "NC_demand_charge_USD"),
    ("OPD peak (kW)",                     "OPD_Peak_kW"),
    ("OPD charge (USD)",                  "OP_demand_charge_USD"),
    ("Missed work (h)",                   "Missed_Work_hour"),
    ("Missed work penalty (USD)",         "Missed_Work_Penalty_USD"),
    ("Terminal SOE shortfall (kWh)",      "Terminal_SOE_Shortfall_kWh"),
    ("Terminal shortfall penalty (USD)",  "Terminal_Shortfall_Penalty_USD"),
    ("MCS transit (h)",                   "MCS_Transit_hour"),
    ("Travel labour (USD)",               "Travel_Labour_USD"),
    ("TOTAL cost (USD)",                  "Total_Cost_USD"),
]

# Exact bar-chart layout from fig08_kpi_summary in 8_ComparisonOutput.jl.
BAR_LABELS = ["Energy", "CO2", "NCD", "OPD", "Missed", "Travel", "Shortfall", "TOTAL"]
BAR_ROWS = ["Total_Energy_Cost_USD", "Total_CO2_Cost_USD", "NC_demand_charge_USD",
            "OP_demand_charge_USD", "Missed_Work_Penalty_USD", "Travel_Labour_USD",
            "Terminal_Shortfall_Penalty_USD", "Total_Cost_USD"]
PEAK_LABELS = ["NCD Peak (kW)", "OPD Peak (kW)"]
PEAK_ROWS = ["NCD_Peak_kW", "OPD_Peak_kW"]


# =============================================================================
# STEP 1 -- read the fixed input constants (no MILP, no Julia, just two CSVs)
# =============================================================================
def read_parameters(parameters_csv: Path) -> dict:
    values = {}
    with open(parameters_csv, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            name = row["Parameter"].strip()
            try:
                values[name] = float(row["Value"])
            except (ValueError, KeyError):
                pass
    required = ["p_digging", "p_loading_swinging", "rho_miss"]
    missing = [r for r in required if r not in values]
    if missing:
        raise ValueError(f"parameters.csv is missing required row(s): {missing}")
    return values


def read_place_hours(place_csv: Path) -> tuple:
    dig_total = 0.0
    load_total = 0.0
    with open(place_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if "hours_digging" not in reader.fieldnames or "hours_loading_swinging" not in reader.fieldnames:
            raise ValueError(
                f"place.csv is missing 'hours_digging'/'hours_loading_swinging' columns; "
                f"found columns: {reader.fieldnames}"
            )
        for row in reader:
            dig_total += float(row["hours_digging"])
            load_total += float(row["hours_loading_swinging"])
    return dig_total, load_total


def compute_avg_work_power_kW(input_dir: Path) -> tuple:
    params = read_parameters(input_dir / "parameters.csv")
    required_dig_h, required_load_h = read_place_hours(input_dir / "place.csv")
    p_digging = params["p_digging"]
    p_loading_swinging = params["p_loading_swinging"]
    rho_miss = params["rho_miss"]

    required_h = required_dig_h + required_load_h
    required_kWh = required_dig_h * p_digging + required_load_h * p_loading_swinging
    avg_work_power_kW = (required_kWh / required_h) if required_h > 1e-9 else (p_digging + p_loading_swinging) / 2

    return avg_work_power_kW, rho_miss, {
        "p_digging": p_digging, "p_loading_swinging": p_loading_swinging,
        "required_dig_h": required_dig_h, "required_load_h": required_load_h,
        "required_h": required_h, "required_kWh": required_kWh,
    }


# =============================================================================
# STEP 2 -- correct one 08_cost_kpi_metrics.csv, return the corrected data
# =============================================================================
def read_metrics_csv(path: Path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    approach_cols = header[1:]
    data = {r[0]: r[1:] for r in rows[1:] if r}
    return header, approach_cols, data


def write_metrics_csv(path: Path, header, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for metric, values in data.items():
            w.writerow([metric] + values)


def correct_metrics(src: Path, avg_work_power_kW: float, rho_miss: float):
    """Returns (header, approach_cols, corrected_data_dict, log_entries) or None
    if this file doesn't have the expected shortfall rows (copied through as-is
    by the caller in that case)."""
    header, approach_cols, data = read_metrics_csv(src)
    needed = ("Terminal_SOE_Shortfall_kWh", "Terminal_Shortfall_Penalty_USD", "Total_Cost_USD")
    if not all(k in data for k in needed):
        return None

    log_entries = []
    new_penalty_row, new_total_row = [], []
    for i, approach in enumerate(approach_cols):
        shortfall_kWh = float(data["Terminal_SOE_Shortfall_kWh"][i])
        old_penalty = float(data["Terminal_Shortfall_Penalty_USD"][i])
        old_total = float(data["Total_Cost_USD"][i])

        new_penalty = rho_miss * (shortfall_kWh / avg_work_power_kW)
        delta = new_penalty - old_penalty
        new_total = old_total + delta

        new_penalty_row.append(f"{new_penalty:.4f}")
        new_total_row.append(f"{new_total:.4f}")
        log_entries.append({
            "approach": approach, "shortfall_kWh": shortfall_kWh,
            "old_penalty": old_penalty, "new_penalty": new_penalty,
            "old_total": old_total, "new_total": new_total, "delta": delta,
        })

    data["Terminal_Shortfall_Penalty_USD"] = new_penalty_row
    data["Total_Cost_USD"] = new_total_row
    return header, approach_cols, data, log_entries


# =============================================================================
# STEP 3 -- regenerate the .html KPI table from corrected data
# =============================================================================
def write_html(dst: Path, approach_cols, data, approach_labels: dict):
    n = len(approach_cols)
    add_delta = n == 2
    lines = []
    lines.append('<!DOCTYPE html><html><head><meta charset="utf-8"><style>')
    lines.append("body{font-family:sans-serif;margin:16px}")
    lines.append("table{border-collapse:collapse;font-size:13px}")
    lines.append("th,td{border:1px solid #ccc;padding:4px 10px;text-align:right}")
    lines.append("th{background:#f4f4f4}")
    lines.append("td:first-child,th:first-child{text-align:left}")
    lines.append("</style></head><body>")
    lines.append("<h2>" + " vs ".join(approach_labels.get(a, a) for a in approach_cols) + "</h2>")
    lines.append(
        "<p><em>Corrected by correct_terminal_shortfall.py -- Terminal shortfall penalty (USD) and "
        "TOTAL cost (USD) recomputed using the required (not realized) work-power conversion rate. "
        "Every other row is unchanged from the original run.</em></p>"
    )
    lines.append("<table><tr><th>Metric</th>")
    for a in approach_cols:
        lines.append(f"<th>{approach_labels.get(a, a)}</th>")
    if add_delta:
        lines.append(f"<th>&Delta; ({approach_labels.get(approach_cols[1], approach_cols[1])} &minus; "
                      f"{approach_labels.get(approach_cols[0], approach_cols[0])})</th>")
    lines.append("</tr>")
    for label, row_name in HTML_ROWS:
        if row_name not in data:
            continue
        vals = [float(v) for v in data[row_name]]
        cells = "".join(f"<td>{v:.3f}</td>" for v in vals)
        row = f"<tr><td>{label}</td>{cells}"
        if add_delta:
            row += f"<td>{vals[1] - vals[0]:.3f}</td>"
        row += "</tr>"
        lines.append(row)
    lines.append("</table></body></html>")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# STEP 4 -- regenerate the 08_kpi_metrics_summary.png bar chart
# =============================================================================
def write_bar_chart(dst: Path, approach_cols, data):
    if not HAVE_MPL:
        print(f"  [!] matplotlib not available -- skipping {dst.name} (pip install matplotlib to enable)")
        return

    fig, (ax_cost, ax_peak) = plt.subplots(2, 1, figsize=(13, 9.5))
    n = len(approach_cols)
    width = 0.8 / n
    x = list(range(len(BAR_LABELS)))
    for ai, approach in enumerate(approach_cols):
        vals = [float(data[r][approach_cols.index(approach)]) if r in data else 0.0 for r in BAR_ROWS]
        offset = (ai - (n - 1) / 2) * width
        ax_cost.bar([xi + offset for xi in x], vals, width=width * 0.92,
                    color=APPROACH_COLORS.get(approach, None), label=approach)
    ax_cost.set_xticks(x)
    ax_cost.set_xticklabels(BAR_LABELS, rotation=20)
    ax_cost.set_ylabel("Cost (USD)")
    ax_cost.set_title("Realised cost components (corrected)")
    ax_cost.legend(loc="upper left")

    xp = list(range(len(PEAK_LABELS)))
    for ai, approach in enumerate(approach_cols):
        vals = [float(data[r][approach_cols.index(approach)]) if r in data else 0.0 for r in PEAK_ROWS]
        offset = (ai - (n - 1) / 2) * width
        ax_peak.bar([xi + offset for xi in xp], vals, width=width * 0.92,
                    color=APPROACH_COLORS.get(approach, None), label=approach)
    ax_peak.set_xticks(xp)
    ax_peak.set_xticklabels(PEAK_LABELS)
    ax_peak.set_ylabel("Power (kW)")
    ax_peak.set_title("Demand peaks")
    ax_peak.legend(loc="upper left")

    fig.suptitle("KPI Metrics Summary (corrected) -- " + " vs ".join(approach_cols))
    fig.tight_layout()
    dst.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dst, dpi=150)
    plt.close(fig)


# =============================================================================
# MAIN -- mirror the WHOLE Output tree: copy everything through untouched,
# except the three file types that actually need correcting.
# =============================================================================
def main():
    input_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(INPUT_DIR)
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(OUTPUT_DIR)
    output_corrected_dir = (
        Path(OUTPUT_CORRECTED_DIR) if OUTPUT_CORRECTED_DIR
        else output_dir.parent / (output_dir.name + "_corrected")
    )
    approach_labels = {"A0": "A0", "A1R": "A1R", "A2R": "A2R"}  # cosmetic only

    print("=" * 88)
    print("TERMINAL SHORTFALL POST-PROCESSING CORRECTION")
    print("=" * 88)
    print(f"Input dir  : {input_dir}")
    print(f"Output dir : {output_dir}  (read-only, never modified)")
    print(f"Writing to : {output_corrected_dir}")
    print(f"matplotlib : {'available' if HAVE_MPL else 'NOT available -- bar charts will be skipped'}")
    print()

    avg_work_power_kW, rho_miss, detail = compute_avg_work_power_kW(input_dir)
    print("Corrected avg_work_power_kW (same for every mode/approach/combo):")
    for k, v in detail.items():
        print(f"  {k:<20} = {v:.4f}")
    print(f"  avg_work_power_kW    = {avg_work_power_kW:.4f} kW   <-- NEW divisor")
    print(f"  rho_miss             = {rho_miss:.4f} $/h")
    print()

    if output_corrected_dir.exists():
        print(f"NOTE: {output_corrected_dir} already exists -- files will be overwritten inside it.")
    output_corrected_dir.mkdir(parents=True, exist_ok=True)

    n_corrected_csv = n_html = n_png = n_copied = 0
    all_log_rows = []

    for src in sorted(output_dir.rglob("*")):
        if src.is_dir():
            continue
        rel = src.relative_to(output_dir)
        dst = output_corrected_dir / rel

        if src.name == "08_cost_kpi_metrics.csv":
            result = correct_metrics(src, avg_work_power_kW, rho_miss)
            if result is None:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                n_copied += 1
                continue
            header, approach_cols, data, log_entries = result
            write_metrics_csv(dst, header, data)
            n_corrected_csv += 1
            mode = rel.parts[0] if len(rel.parts) > 0 else ""
            combo = rel.parts[1] if len(rel.parts) > 1 else ""
            for e in log_entries:
                e["mode"], e["combo"] = mode, combo
                all_log_rows.append(e)

            # Regenerate the sibling .html and bar-chart PNG in the SAME folder,
            # using this same corrected data (they're 1:1 with the CSV).
            for html_src in src.parent.glob("*.html"):
                write_html(output_corrected_dir / html_src.relative_to(output_dir), approach_cols, data, approach_labels)
                n_html += 1
            bar_png = src.parent / "08_kpi_metrics_summary.png"
            if bar_png.exists():
                write_bar_chart(output_corrected_dir / bar_png.relative_to(output_dir), approach_cols, data)
                n_png += 1
            continue

        if src.suffix == ".html" or src.name == "08_kpi_metrics_summary.png":
            # Already regenerated above, alongside its 08_cost_kpi_metrics.csv sibling.
            continue

        # Everything else: copy through byte-identical.
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        n_copied += 1

    print(f"Corrected {n_corrected_csv} cost-KPI CSV file(s).")
    print(f"Regenerated {n_html} HTML file(s) and {n_png} bar-chart PNG(s).")
    print(f"Copied {n_copied} other file(s) through untouched (including the top-level")
    print("mode_sweep_kpi_summary.csv and run_log.txt -- confirmed unaffected by this fix).")
    print()

    print("=" * 88)
    print("OLD vs NEW  (only Terminal_Shortfall_Penalty_USD and Total_Cost_USD change)")
    print("=" * 88)
    print(f"{'mode':<13}{'combo':<12}{'app':<5}{'shortfall_kWh':>14}{'old_penalty':>13}"
          f"{'new_penalty':>13}{'old_total':>11}{'new_total':>11}{'delta':>9}")
    for r in all_log_rows:
        print(f"{r['mode']:<13}{r['combo']:<12}{r['approach']:<5}"
              f"{r['shortfall_kWh']:>14.3f}{r['old_penalty']:>13.2f}{r['new_penalty']:>13.2f}"
              f"{r['old_total']:>11.2f}{r['new_total']:>11.2f}{r['delta']:>9.2f}")
    print("=" * 88)
    print(f"\nDone. Corrected tree written to: {output_corrected_dir}")
    print("Nothing under the original Output/ folder was modified.")


if __name__ == "__main__":
    main()
