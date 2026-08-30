#!/usr/bin/env python3
# =============================================================================
# rebuild_sweep_summary.py
# -----------------------------------------------------------------------------
# Run this ONCE all 5 manual mode scripts (abc.jl / near_mean.jl / high.jl /
# low.jl / spread_wide.jl) have finished. It produces the two things the real
# sweep driver (7_Comparison_main_ShrinkingOnlyVersion_Sweep.jl) would have
# auto-generated but the manual, unwrapped runs don't:
#
#   1. Output/mode_sweep_kpi_summary.csv
#        Rebuilt from each mode's own A0_A1S_A2S/08_cost_kpi_metrics.csv.
#        Column mapping verified against the real sweep driver's own code:
#        `total_cost_USD` here is ENERGY-ONLY cost (Total_Energy_Cost_USD),
#        NOT the full aggregate Total_Cost_USD -- that's a real, confirmed
#        quirk of the original file's column naming, not a mistake here.
#
#   2. Downloads/rebuilt_run_log.txt
#        Pulls ONLY the three "Running Approach N ... done in Xs" summary
#        blocks out of each mode's raw Tee-Object log (log_manual_<mode>.txt)
#        -- discarding the huge HiGHS branch-and-bound trace text -- and
#        stitches them together with the same "# MODE: :xxx #" section
#        headers the real run_log.txt uses. It does NOT fabricate the
#        bracketed [HH:MM:SS] timing lines the real sweep prints, since the
#        manual runs never recorded real per-step timestamps to reconstruct
#        those from -- inventing fake ones would be less honest than leaving
#        them out.
#
# USAGE:
#   Edit the paths in CONFIG below if yours differ, then:
#   python rebuild_sweep_summary.py
# =============================================================================

import csv
import re
from pathlib import Path

# ---- CONFIG -- edit if your paths differ ----
COMPARISON_OUT = Path(r"C:\Users\shubh\Desktop\MPC\Comparison_A0_A1_A2\Output")
DOWNLOADS = Path(r"C:\Users\shubh\Downloads")

MODES = ["normal", "near_mean", "high", "low", "spread_wide"]
LOG_FILES = {
    "normal":       DOWNLOADS / "log_manual_normal.txt",
    "near_mean":    DOWNLOADS / "log_manual_near_mean.txt",
    "high":         DOWNLOADS / "log_manual_high.txt",
    "low":          DOWNLOADS / "log_manual_low.txt",
    "spread_wide":  DOWNLOADS / "log_manual_spread_wide.txt",
}

SUMMARY_OUT = COMPARISON_OUT / "mode_sweep_kpi_summary.csv"
LOG_OUT = COMPARISON_OUT / "run_log.txt"

APPROACHES = ["A0", "A1S", "A2S"]

# mode_sweep_kpi_summary.csv column -> 08_cost_kpi_metrics.csv row name.
# total_cost_USD really is energy-only here -- confirmed against the real
# sweep driver's own res.total_cost field, which is sum(grid_kW .* price) *
# delta_T, not the full aggregate. Kept exactly consistent with the original.
SUMMARY_COLS = [
    ("total_energy_kWh", "Total_Grid_Energy_kWh"),
    ("total_cost_USD",   "Total_Energy_Cost_USD"),
    ("total_co2_kg",     "Total_CO2_Emissions_kg"),
    ("nc_peak_kW",       "NCD_Peak_kW"),
    ("op_peak_kW",       "OPD_Peak_kW"),
    ("missed_h",         "Missed_Work_hour"),
    ("shortfall_kWh",    "Terminal_SOE_Shortfall_kWh"),
]


# =============================================================================
# PART 1 -- rebuild mode_sweep_kpi_summary.csv
# =============================================================================
def read_kpi_csv(path: Path) -> dict:
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    approach_cols = header[1:]
    data = {r[0]: r[1:] for r in rows[1:] if r}
    return approach_cols, data


def build_summary():
    out_rows = []
    missing = []
    for mode in MODES:
        kpi_path = COMPARISON_OUT / mode / "A0_A1S_A2S" / "08_cost_kpi_metrics.csv"
        if not kpi_path.exists():
            missing.append(str(kpi_path))
            continue
        approach_cols, data = read_kpi_csv(kpi_path)
        for approach in APPROACHES:
            if approach not in approach_cols:
                continue
            i = approach_cols.index(approach)
            row = {"mode": mode, "approach": approach}
            for summary_key, metric_name in SUMMARY_COLS:
                row[summary_key] = data[metric_name][i] if metric_name in data else ""
            out_rows.append(row)

    if missing:
        print("WARNING: could not find these expected files (skipped):")
        for m in missing:
            print("  ", m)

    if not out_rows:
        print("No data found at all -- check COMPARISON_OUT path. Nothing written.")
        return

    with open(SUMMARY_OUT, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["mode", "approach"] + [k for k, _ in SUMMARY_COLS]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    print(f"Wrote {len(out_rows)} rows to: {SUMMARY_OUT}")


# =============================================================================
# PART 2 -- rebuild a clean run_log.txt (just the 3 approach summary blocks
# per mode, no HiGHS branch-and-bound noise)
# =============================================================================
BLOCK_START = re.compile(r"^Running Approach \d")
DONE_LINE = re.compile(r"done in .* \(96 plant realizations")
INDENTED = re.compile(r"^\s+\S")


def extract_approach_blocks(text: str) -> list:
    """Returns a list of block strings, each starting at a 'Running Approach N'
    line. Phase 1: keep the start line plus any immediately-following
    indented metadata lines. Phase 2: skip forward past whatever comes next
    (this is where the huge HiGHS branch-and-bound trace lives, if present)
    until finding the unindented 'done in ... plant realizations' summary
    line, and keep that. Phase 3: keep any indented metadata lines that
    follow it. Searching past unknown noise in phase 2 (rather than stopping
    at the first unindented line) is what's needed here, since the HiGHS
    trace itself is unindented and would otherwise look like a stop signal."""
    lines = text.splitlines()
    blocks = []
    i = 0
    n = len(lines)
    while i < n:
        if BLOCK_START.match(lines[i]):
            block_lines = [lines[i]]
            j = i + 1
            while j < n and INDENTED.match(lines[j]):
                block_lines.append(lines[j])
                j += 1
            while j < n and not DONE_LINE.search(lines[j]) and not BLOCK_START.match(lines[j]):
                j += 1
            if j < n and DONE_LINE.search(lines[j]):
                block_lines.append(lines[j])
                j += 1
                while j < n and INDENTED.match(lines[j]):
                    block_lines.append(lines[j])
                    j += 1
            while block_lines and block_lines[-1].strip() == "":
                block_lines.pop()
            blocks.append("\n".join(block_lines))
            i = j
        else:
            i += 1
    return blocks


def read_log_text(path: Path) -> str:
    """Tee-Object saves PowerShell transcripts as UTF-16 (with a BOM); plain
    println-redirected output is usually UTF-8. Sniff the BOM and decode
    accordingly rather than assuming one or the other."""
    raw = path.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    return raw.decode("utf-8", errors="replace")


def build_log():
    sections = []
    any_found = False
    for mode in MODES:
        log_path = LOG_FILES[mode]
        header = (
            "\n" + "#" * 78 + "\n"
            f"# MODE: :{mode}\n"
            + "#" * 78
        )
        if not log_path.exists():
            sections.append(header + f"\n[log file not found: {log_path}]")
            continue
        text = read_log_text(log_path)
        blocks = extract_approach_blocks(text)
        if not blocks:
            sections.append(header + "\n[no 'Running Approach' blocks found in this log]")
            continue
        any_found = True
        incomplete_note = ""
        if not any("done in" in b for b in blocks[-1:]):
            incomplete_note = "\n[NOTE: last block has no 'done in' line -- this mode's run may not have finished when the log was captured]"
        sections.append(header + "\n\n" + "\n\n".join(blocks) + incomplete_note)

    if LOG_OUT.exists():
        print(f"NOTE: {LOG_OUT} already exists (e.g. from the earlier killed sweep) -- overwriting it.")

    with open(LOG_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(sections).lstrip() + "\n")

    if any_found:
        print(f"Wrote reconstructed log to: {LOG_OUT}")
    else:
        print("WARNING: no approach blocks found in ANY log file -- check LOG_FILES paths.")
        print(f"(Still wrote a placeholder file to: {LOG_OUT})")


if __name__ == "__main__":
    print("=" * 78)
    print("Rebuilding mode_sweep_kpi_summary.csv ...")
    print("=" * 78)
    build_summary()
    print()
    print("=" * 78)
    print("Rebuilding run_log.txt (approach summary blocks only) ...")
    print("=" * 78)
    build_log()
