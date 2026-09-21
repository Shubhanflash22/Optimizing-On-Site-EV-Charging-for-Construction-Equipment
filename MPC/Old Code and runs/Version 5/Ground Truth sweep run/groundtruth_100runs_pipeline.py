"""
groundtruth_100runs_pipeline.py
===============================
Single script that does the whole ground-truth post-processing in one go:

  STEP 1  Read  Output_GroundTruth_100Runs/<label>/live_data/A0_A1S_A2S/A0_vs_A1S_vs_A2S.html
          for all 100 runs and compile the two Excel workbooks
              GroundTruth_100Runs_KPI_Matrix.xlsx
              GroundTruth_100Runs_TotalCost_Analysis.xlsx

  STEP 2  Re-open those freshly written Excel files and use THEM (not the HTML)
          as the only data source for all 12 matplotlib figures (PNG).

Everything is saved in OUT_DIR.

Usage
-----
    python groundtruth_100runs_pipeline.py [OUTPUT_GT_DIR] [OUT_DIR]

Defaults
--------
    OUTPUT_GT_DIR = C:\\Users\\shubh\\Desktop\\MPC\\Actual Run Data\\Ground Truth 100 runs data\\Output_GroundTruth_100Runs
    OUT_DIR       = C:\\Users\\shubh\\Desktop\\MPC\\Actual Run Data\\Ground Truth 100 runs data
"""

import os, sys, re, math
from pathlib import Path
from html.parser import HTMLParser
from statistics import mean, median, stdev

import numpy as np
import matplotlib
matplotlib.use("Agg")                          # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

import openpyxl
from openpyxl.styles import (Font, PatternFill, Alignment, Border, Side)
from openpyxl.utils import get_column_letter
from openpyxl.chart import LineChart, BarChart, Reference
from openpyxl.chart.series import DataPoint

# ###########################################################################
#   PART A — COMPILE HTML -> EXCEL   (unchanged from compile_groundtruth_100runs.py)
# ###########################################################################

# ---------------------------------------------------------------------------
# 1. Run definitions — 100 seeds, all mode=live_data
# ---------------------------------------------------------------------------
RUNS = [
    dict(label=f"{i:03d}_GroundTruth_seed{i}_1200s", mode="live_data", seed=i)
    for i in range(1, 101)
]

APPROACH_LABELS = ["A0 (OS)", "A1S (CE)", "A2S (SB)"]   # display names
APPROACH_SHORT  = ["A0", "A1S", "A2S"]                   # short names

EXPECTED_METRICS = [
    "Grid energy (kWh)",
    "Energy cost (USD)",
    "CO2 emissions (kg)",
    "CO2 cost (USD)",
    "NCD peak (kW)",
    "NCD charge (USD)",
    "OPD peak (kW)",
    "OPD charge (USD)",
    "Missed work (h)",
    "Missed work penalty (USD)",
    "Terminal SOE shortfall (kWh)",
    "Terminal shortfall penalty (USD)",
    "MCS transit (h)",
    "Travel labour (USD)",
    "TOTAL cost (USD)",
]

TOTAL_METRIC = "TOTAL cost (USD)"

# ---------------------------------------------------------------------------
# 2. HTML parser
# ---------------------------------------------------------------------------
class TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_table = False
        self.in_cell  = False
        self.rows: list[list[str]] = []
        self._row: list[str] = []
        self._cell_text = ""

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.in_table = True
        elif tag == "tr" and self.in_table:
            self._row = []
        elif tag in ("td", "th") and self.in_table:
            self.in_cell = True
            self._cell_text = ""

    def handle_endtag(self, tag):
        if tag == "table":
            self.in_table = False
        elif tag == "tr" and self.in_table:
            if self._row:
                self.rows.append(self._row)
        elif tag in ("td", "th") and self.in_table:
            self.in_cell = False
            self._row.append(self._cell_text.strip())

    def handle_data(self, data):
        if self.in_cell:
            self._cell_text += data


def parse_html(path: Path) -> dict[str, list[str]]:
    """Returns {metric_name: [A0_value, A1S_value, A2S_value]}"""
    parser = TableParser()
    parser.feed(path.read_text(encoding="utf-8"))
    result = {}
    for row in parser.rows[1:]:
        if len(row) >= 4:
            result[row[0]] = row[1:4]
    return result


def html_path(root: Path, run: dict) -> Path:
    return root / run["label"] / run["mode"] / "A0_A1S_A2S" / "A0_vs_A1S_vs_A2S.html"


# ---------------------------------------------------------------------------
# 3. Styling helpers
# ---------------------------------------------------------------------------
FONT_NAME = "Calibri"

# Colour palette
C_DARK_BLUE   = "1F4E79"
C_MED_BLUE    = "2E75B6"
C_LIGHT_BLUE  = "D6E4F0"
C_A0          = "EBF3FB"
C_A1S         = "D6E4F0"
C_A2S         = "BDD7EE"
C_TOTAL       = "FCE4D6"
C_MISSING     = "FFE7E7"
C_GREEN       = "E2EFDA"
C_ORANGE      = "FCE4D6"
C_RED_LIGHT   = "FFDDC1"
C_HEADER_TXT  = "FFFFFF"
C_GOLD        = "FFD700"

FILL_DARK_BLUE  = PatternFill("solid", fgColor=C_DARK_BLUE)
FILL_MED_BLUE   = PatternFill("solid", fgColor=C_MED_BLUE)
FILL_LIGHT_BLUE = PatternFill("solid", fgColor=C_LIGHT_BLUE)
FILL_A0         = PatternFill("solid", fgColor=C_A0)
FILL_A1S        = PatternFill("solid", fgColor=C_A1S)
FILL_A2S        = PatternFill("solid", fgColor=C_A2S)
FILL_TOTAL      = PatternFill("solid", fgColor=C_TOTAL)
FILL_MISSING    = PatternFill("solid", fgColor=C_MISSING)
FILL_GREEN      = PatternFill("solid", fgColor=C_GREEN)
FILL_ORANGE     = PatternFill("solid", fgColor=C_ORANGE)
FILL_RED_LIGHT  = PatternFill("solid", fgColor=C_RED_LIGHT)
FILL_GOLD       = PatternFill("solid", fgColor=C_GOLD)

THIN = Side(style="thin",   color="AAAAAA")
MED  = Side(style="medium", color="666666")

def thin_border():
    return Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

def med_border():
    return Border(left=MED, right=MED, top=MED, bottom=MED)

def sc(cell, *, bold=False, italic=False, size=10, color="000000",
       fill=None, halign="center", valign="center",
       border=None, wrap=False, num_fmt=None):
    cell.font = Font(name=FONT_NAME, bold=bold, italic=italic,
                     size=size, color=color)
    cell.alignment = Alignment(horizontal=halign, vertical=valign,
                                wrap_text=wrap)
    if fill:   cell.fill   = fill
    if border: cell.border = border
    if num_fmt: cell.number_format = num_fmt


def hdr(cell, value, *, level="top", halign="center"):
    """Quick-style a header cell."""
    cell.value = value
    fill  = FILL_DARK_BLUE if level == "top" else FILL_MED_BLUE
    sc(cell, bold=True, size=10, color=C_HEADER_TXT,
       fill=fill, halign=halign, border=thin_border())


def num(cell, value, *, fill=None, bold=False, fmt="#,##0.000"):
    cell.value = value
    sc(cell, bold=bold, size=10, fill=fill or PatternFill(),
       border=thin_border(), num_fmt=fmt, halign="center")


# ---------------------------------------------------------------------------
# 4. Load all run data
# ---------------------------------------------------------------------------
def load_all(root: Path) -> list[dict | None]:
    run_data = []
    for run in RUNS:
        hp = html_path(root, run)
        if hp.exists():
            try:
                run_data.append(parse_html(hp))
            except Exception as e:
                print(f"  [WARN] parse error {hp}: {e}")
                run_data.append(None)
        else:
            print(f"  [INFO] missing: {hp}")
            run_data.append(None)
    return run_data


def safe_float(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# 5. FILE 1 — KPI Matrix
# ---------------------------------------------------------------------------
def build_kpi_matrix(root: Path, out_path: Path, run_data: list):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "KPI Matrix"

    n_runs = len(RUNS)
    n_ap   = 3   # A0, A1S, A2S

    def run_col_start(idx):
        return 2 + idx * n_ap   # col B = 2

    # --- Row 1: corner + run labels ---
    c = ws.cell(row=1, column=1, value="Run / Metric")
    sc(c, bold=True, size=11, color=C_HEADER_TXT,
       fill=FILL_DARK_BLUE, border=thin_border())

    for i, (run, data) in enumerate(zip(RUNS, run_data)):
        col_s = run_col_start(i)
        col_e = col_s + n_ap - 1
        ws.merge_cells(start_row=1, start_column=col_s,
                       end_row=1,   end_column=col_e)
        cell = ws.cell(row=1, column=col_s)
        if data is not None:
            cell.value = f"Seed {run['seed']}"
            sc(cell, bold=True, size=9, color=C_HEADER_TXT,
               fill=FILL_DARK_BLUE, halign="center", border=thin_border())
        else:
            cell.value = f"MISSING\nSeed {run['seed']}"
            sc(cell, bold=True, italic=True, size=8, color="C00000",
               fill=FILL_MISSING, halign="center", border=thin_border(), wrap=True)

    # --- Row 2: Metric | A0 A1S A2S | A0 A1S A2S | … ---
    c = ws.cell(row=2, column=1, value="Metric")
    sc(c, bold=True, size=10, color=C_HEADER_TXT,
       fill=FILL_MED_BLUE, border=thin_border())

    AP_FILLS = [FILL_A0, FILL_A1S, FILL_A2S]
    for i in range(n_runs):
        col_s = run_col_start(i)
        for j, ah in enumerate(APPROACH_SHORT):
            c = ws.cell(row=2, column=col_s + j, value=ah)
            sc(c, bold=True, size=9, color=C_HEADER_TXT,
               fill=FILL_MED_BLUE, border=thin_border())

    # --- Rows 3+: metrics ---
    for m_idx, metric in enumerate(EXPECTED_METRICS):
        row = 3 + m_idx
        is_total = "TOTAL" in metric.upper()

        lc = ws.cell(row=row, column=1, value=metric)
        sc(lc, bold=is_total, size=10, color="1F3864",
           fill=(FILL_TOTAL if is_total else FILL_LIGHT_BLUE),
           halign="left", border=thin_border())

        for i, (run, data) in enumerate(zip(RUNS, run_data)):
            col_s = run_col_start(i)
            for j in range(n_ap):
                cell = ws.cell(row=row, column=col_s + j)
                if data is None:
                    cell.value = ""
                    sc(cell, size=10, fill=FILL_MISSING, border=thin_border())
                else:
                    vals = data.get(metric, ["", "", ""])
                    raw  = vals[j] if j < len(vals) else ""
                    fval = safe_float(raw)
                    cell.value = fval if fval is not None else raw
                    fill = FILL_TOTAL if is_total else AP_FILLS[j]
                    sc(cell, bold=is_total, size=9, fill=fill,
                       border=thin_border(), num_fmt="#,##0.000")

    # --- Column widths ---
    ws.column_dimensions["A"].width = 34
    for i in range(n_runs):
        col_s = run_col_start(i)
        for j in range(n_ap):
            ws.column_dimensions[get_column_letter(col_s + j)].width = 9

    ws.row_dimensions[1].height = 28
    ws.row_dimensions[2].height = 16
    ws.freeze_panes = "B3"

    # --- Legend sheet ---
    ws_leg = wb.create_sheet("Legend")
    leg_rows = [
        ("Abbreviation", "Full name", "Description"),
        ("A0 / OS", "One-Shot Naïve Optimization",
         "Approach 0 — single-shot optimisation before the day starts"),
        ("A1S / CE", "Certainty Equivalent Shrinking MPC",
         "Approach 1 — shrinking-horizon MPC, deterministic (expected values)"),
        ("A2S / SB", "Scenario-Based Shrinking MPC",
         "Approach 2 — shrinking-horizon stochastic / scenario-based MPC"),
        ("", "", ""),
        ("MISSING", "", "HTML result file not found; cells left blank"),
    ]
    for r_idx, leg_row in enumerate(leg_rows, 1):
        for c_idx, val in enumerate(leg_row, 1):
            cell = ws_leg.cell(row=r_idx, column=c_idx, value=val)
            sc(cell, bold=(r_idx == 1), size=10,
               fill=(FILL_DARK_BLUE if r_idx == 1 else None),
               color=(C_HEADER_TXT if r_idx == 1 else "000000"),
               halign="left", border=thin_border())
    ws_leg.column_dimensions["A"].width = 14
    ws_leg.column_dimensions["B"].width = 38
    ws_leg.column_dimensions["C"].width = 62

    wb.save(out_path)
    print(f"  [OK] KPI Matrix saved: {out_path}")


# ---------------------------------------------------------------------------
# 6. FILE 2 — Total Cost Analysis
# ---------------------------------------------------------------------------
def build_cost_analysis(root: Path, out_path: Path, run_data: list):
    wb = openpyxl.Workbook()

    # -----------------------------------------------------------------------
    # Sheet 1: Total Cost row-by-row
    # -----------------------------------------------------------------------
    ws = wb.active
    ws.title = "Total Cost"

    # Header row
    headers = [
        "Seed", "Run Label",
        "A0 Total Cost (USD)", "A1S Total Cost (USD)", "A2S Total Cost (USD)",
        "A1S vs A0 Saving (USD)", "A2S vs A0 Saving (USD)", "A1S vs A2S Diff (USD)",
        "Best Approach", "A2S < A1S < A0?", "Chain A0≥A1S≥A2S Holds?",
        "Notes"
    ]
    for col, h in enumerate(headers, 1):
        c = ws.cell(row=1, column=col, value=h)
        sc(c, bold=True, size=10, color=C_HEADER_TXT,
           fill=FILL_DARK_BLUE, halign="center", border=thin_border(), wrap=True)

    col_widths = [7, 38, 20, 20, 20, 20, 20, 20, 16, 18, 22, 30]
    for i, w in enumerate(col_widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.row_dimensions[1].height = 30

    # Data rows
    present_rows = []   # (seed, a0, a1s, a2s) for stats
    for i, (run, data) in enumerate(zip(RUNS, run_data)):
        row = i + 2
        seed = run["seed"]

        if data is None:
            ws.cell(row=row, column=1, value=seed)
            ws.cell(row=row, column=2, value=run["label"])
            for col in range(3, len(headers) + 1):
                c = ws.cell(row=row, column=col, value="MISSING")
                sc(c, italic=True, size=9, color="C00000",
                   fill=FILL_MISSING, border=thin_border())
            continue

        vals  = data.get(TOTAL_METRIC, ["", "", ""])
        a0    = safe_float(vals[0]) if len(vals) > 0 else None
        a1s   = safe_float(vals[1]) if len(vals) > 1 else None
        a2s   = safe_float(vals[2]) if len(vals) > 2 else None

        if a0 is not None and a1s is not None and a2s is not None:
            present_rows.append((seed, a0, a1s, a2s))

        d_a1_a0 = (a0 - a1s)   if (a0 and a1s) else None
        d_a2_a0 = (a0 - a2s)   if (a0 and a2s) else None
        d_a1_a2 = (a1s - a2s)  if (a1s and a2s) else None

        # Best approach
        costs = {"A0": a0, "A1S": a1s, "A2S": a2s}
        valid = {k: v for k, v in costs.items() if v is not None}
        best = min(valid, key=lambda k: valid[k]) if valid else "?"

        # Is A2S < A1S < A0?
        strict_chain = (a0 is not None and a1s is not None and a2s is not None
                        and a2s < a1s < a0)
        # Does A0 >= A1S >= A2S hold? (non-strict)
        chain_holds = (a0 is not None and a1s is not None and a2s is not None
                       and a0 >= a1s >= a2s)

        notes = []
        if a1s is not None and a0 is not None and a1s > a0:
            notes.append("A1S worse than A0")
        if a2s is not None and a1s is not None and a2s > a1s:
            notes.append("A2S worse than A1S")
        if a2s is not None and a0 is not None and a2s > a0:
            notes.append("A2S worse than A0")

        row_fill = FILL_GREEN if chain_holds else FILL_RED_LIGHT

        def put(col, val, fmt="#,##0.000", bold=False, fill=None):
            c = ws.cell(row=row, column=col, value=val)
            fval = safe_float(val) if isinstance(val, str) else val
            if fval is not None and isinstance(val, (int, float)):
                sc(c, bold=bold, size=10, fill=fill or row_fill,
                   border=thin_border(), num_fmt=fmt, halign="center")
            else:
                sc(c, bold=bold, size=10, fill=fill or row_fill,
                   border=thin_border(), halign="center")

        ws.cell(row=row, column=1, value=seed)
        sc(ws.cell(row=row, column=1), bold=True, size=10,
           fill=row_fill, border=thin_border(), halign="center")

        ws.cell(row=row, column=2, value=run["label"])
        sc(ws.cell(row=row, column=2), size=9, fill=row_fill,
           border=thin_border(), halign="left")

        # A0, A1S, A2S cost cells
        for col, val, ap_fill in [(3, a0,  FILL_A0),
                                   (4, a1s, FILL_A1S),
                                   (5, a2s, FILL_A2S)]:
            c = ws.cell(row=row, column=col, value=val)
            sc(c, bold=True, size=10, fill=ap_fill,
               border=thin_border(), num_fmt="#,##0.000", halign="center")

        # Saving columns — green if positive (saving), orange if negative
        for col, val in [(6, d_a1_a0), (7, d_a2_a0), (8, d_a1_a2)]:
            c = ws.cell(row=row, column=col, value=val)
            f = FILL_GREEN if (val is not None and val > 0) else FILL_ORANGE
            sc(c, size=10, fill=f, border=thin_border(),
               num_fmt="+#,##0.000;-#,##0.000;0.000", halign="center")

        # Best approach
        best_fill = {"A0": FILL_A0, "A1S": FILL_A1S, "A2S": FILL_A2S}.get(best, FILL_LIGHT_BLUE)
        c = ws.cell(row=row, column=9, value=best)
        sc(c, bold=True, size=10, fill=best_fill,
           border=thin_border(), halign="center")

        c = ws.cell(row=row, column=10,
                    value="YES" if strict_chain else "NO")
        sc(c, bold=True, size=10,
           fill=(FILL_GREEN if strict_chain else FILL_RED_LIGHT),
           border=thin_border(), halign="center")

        c = ws.cell(row=row, column=11,
                    value="YES" if chain_holds else "NO")
        sc(c, bold=True, size=10,
           fill=(FILL_GREEN if chain_holds else FILL_RED_LIGHT),
           border=thin_border(), halign="center")

        c = ws.cell(row=row, column=12, value="; ".join(notes) if notes else "OK")
        sc(c, size=9, fill=(FILL_ORANGE if notes else FILL_GREEN),
           border=thin_border(), halign="left")

    # Freeze header
    ws.freeze_panes = "A2"

    # -----------------------------------------------------------------------
    # Sheet 2: Cost Summary statistics
    # -----------------------------------------------------------------------
    ws2 = wb.create_sheet("Cost Summary")

    if present_rows:
        seeds, a0_vals, a1s_vals, a2s_vals = zip(*present_rows)
    else:
        seeds = a0_vals = a1s_vals = a2s_vals = []

    def stats(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return [None] * 7
        q = sorted(vals)
        n = len(q)
        q1 = q[n // 4]
        q3 = q[(3 * n) // 4]
        return [len(vals), mean(vals), median(vals),
                stdev(vals) if len(vals) > 1 else 0,
                min(vals), q1, max(vals)]

    stat_labels = ["Count", "Mean", "Median", "Std Dev", "Min", "Q1 (25th pct)", "Max"]
    ap_data = [
        ("A0 (One-Shot)", list(a0_vals),  FILL_A0),
        ("A1S (CE MPC)",  list(a1s_vals), FILL_A1S),
        ("A2S (SB MPC)",  list(a2s_vals), FILL_A2S),
    ]

    # Section title
    ws2.merge_cells("A1:D1")
    c = ws2.cell(row=1, column=1,
                 value="Total Cost (USD) — Summary Statistics across 100 Seeds")
    sc(c, bold=True, size=12, color=C_HEADER_TXT,
       fill=FILL_DARK_BLUE, halign="left", border=thin_border())

    hdr(ws2.cell(row=2, column=1), "Statistic", level="sub", halign="left")
    for col, (ap_name, _, ap_fill) in enumerate(ap_data, 2):
        c = ws2.cell(row=2, column=col, value=ap_name)
        sc(c, bold=True, size=10, color=C_HEADER_TXT,
           fill=FILL_MED_BLUE, border=thin_border())

    for r, (label, *_) in enumerate([(sl,) for sl in stat_labels], 3):
        sl = stat_labels[r - 3]
        c = ws2.cell(row=r, column=1, value=sl)
        sc(c, bold=True, size=10, fill=FILL_LIGHT_BLUE,
           halign="left", border=thin_border())
        for col, (_, vals, ap_fill) in enumerate(ap_data, 2):
            sv = stats(vals)
            val = sv[r - 3]
            cell = ws2.cell(row=r, column=col, value=val)
            sc(cell, size=10, fill=ap_fill, border=thin_border(),
               num_fmt="#,##0.000", halign="center")

    # Win counts section
    win_row_start = 3 + len(stat_labels) + 2
    ws2.merge_cells(start_row=win_row_start, start_column=1,
                    end_row=win_row_start, end_column=4)
    c = ws2.cell(row=win_row_start, column=1,
                 value="Approach Win Counts (lowest total cost)")
    sc(c, bold=True, size=11, color=C_HEADER_TXT,
       fill=FILL_DARK_BLUE, halign="left", border=thin_border())

    win_counts = {"A0": 0, "A1S": 0, "A2S": 0, "Tie": 0}
    chain_hold_count  = 0
    strict_chain_count = 0
    for _, a0, a1s, a2s in present_rows:
        best_val = min(a0, a1s, a2s)
        winners = [k for k, v in [("A0", a0), ("A1S", a1s), ("A2S", a2s)]
                   if abs(v - best_val) < 1e-6]
        if len(winners) > 1:
            win_counts["Tie"] += 1
        else:
            win_counts[winners[0]] += 1
        if a0 >= a1s >= a2s:
            chain_hold_count += 1
        if a2s < a1s < a0:
            strict_chain_count += 1

    win_headers = ["Approach", "Wins", "% of runs", "Fill"]
    hdr(ws2.cell(row=win_row_start + 1, column=1), "Approach", level="sub", halign="left")
    hdr(ws2.cell(row=win_row_start + 1, column=2), "Wins",      level="sub")
    hdr(ws2.cell(row=win_row_start + 1, column=3), "% of Runs", level="sub")

    n_present = len(present_rows)
    win_display = [
        ("A0 (One-Shot)", win_counts["A0"],  FILL_A0),
        ("A1S (CE MPC)",  win_counts["A1S"], FILL_A1S),
        ("A2S (SB MPC)",  win_counts["A2S"], FILL_A2S),
        ("Tie",           win_counts["Tie"], FILL_LIGHT_BLUE),
    ]
    for r_off, (ap_name, cnt, ap_fill) in enumerate(win_display, 2):
        row = win_row_start + r_off
        c = ws2.cell(row=row, column=1, value=ap_name)
        sc(c, bold=True, size=10, fill=ap_fill, halign="left", border=thin_border())
        c = ws2.cell(row=row, column=2, value=cnt)
        sc(c, size=10, fill=ap_fill, border=thin_border(),
           num_fmt="0", halign="center")
        pct = (cnt / n_present * 100) if n_present else 0
        c = ws2.cell(row=row, column=3, value=pct)
        sc(c, size=10, fill=ap_fill, border=thin_border(),
           num_fmt="0.0%", halign="center")

    # Chain analysis
    chain_row = win_row_start + len(win_display) + 3
    ws2.merge_cells(start_row=chain_row, start_column=1,
                    end_row=chain_row, end_column=4)
    c = ws2.cell(row=chain_row, column=1, value="Chain Analysis: A0 ≥ A1S ≥ A2S")
    sc(c, bold=True, size=11, color=C_HEADER_TXT,
       fill=FILL_DARK_BLUE, halign="left", border=thin_border())

    chain_rows_data = [
        ("Strict A2S < A1S < A0 (all three ranked)", strict_chain_count,
         strict_chain_count / n_present * 100 if n_present else 0),
        ("Non-strict A0 ≥ A1S ≥ A2S (chain holds)", chain_hold_count,
         chain_hold_count / n_present * 100 if n_present else 0),
        ("Chain broken (at least one inversion)",
         n_present - chain_hold_count,
         (n_present - chain_hold_count) / n_present * 100 if n_present else 0),
    ]
    hdr(ws2.cell(row=chain_row + 1, column=1), "Condition", level="sub", halign="left")
    hdr(ws2.cell(row=chain_row + 1, column=2), "Count",     level="sub")
    hdr(ws2.cell(row=chain_row + 1, column=3), "% of Runs", level="sub")

    for r_off, (label, cnt, pct) in enumerate(chain_rows_data, 2):
        row = chain_row + r_off
        f = FILL_GREEN if "holds" in label or "Strict" in label else FILL_RED_LIGHT
        c = ws2.cell(row=row, column=1, value=label)
        sc(c, size=10, fill=f, halign="left", border=thin_border())
        c = ws2.cell(row=row, column=2, value=cnt)
        sc(c, size=10, fill=f, border=thin_border(), num_fmt="0", halign="center")
        c = ws2.cell(row=row, column=3, value=pct / 100)
        sc(c, size=10, fill=f, border=thin_border(), num_fmt="0.0%", halign="center")

    ws2.column_dimensions["A"].width = 44
    ws2.column_dimensions["B"].width = 18
    ws2.column_dimensions["C"].width = 14
    ws2.column_dimensions["D"].width = 14

    # -----------------------------------------------------------------------
    # Sheet 3: Charts data + embedded charts
    # -----------------------------------------------------------------------
    ws3 = wb.create_sheet("Charts")

    # Title
    ws3.merge_cells("A1:F1")
    c = ws3.cell(row=1, column=1,
                 value="Ground Truth 100 Runs — Total Cost Visualisations")
    sc(c, bold=True, size=13, color=C_HEADER_TXT,
       fill=FILL_DARK_BLUE, halign="left", border=thin_border())

    # ---- Data table for the line chart (cols A-D, rows 3 onward) ----
    data_start_row = 3
    hdr(ws3.cell(row=data_start_row, column=1), "Seed",         level="sub", halign="left")
    hdr(ws3.cell(row=data_start_row, column=2), "A0 Cost (USD)", level="sub")
    hdr(ws3.cell(row=data_start_row, column=3), "A1S Cost (USD)", level="sub")
    hdr(ws3.cell(row=data_start_row, column=4), "A2S Cost (USD)", level="sub")

    chart_data_rows = []
    for i, (run, data) in enumerate(zip(RUNS, run_data)):
        row = data_start_row + 1 + i
        seed = run["seed"]
        if data is not None:
            vals = data.get(TOTAL_METRIC, ["", "", ""])
            a0  = safe_float(vals[0]) if len(vals) > 0 else None
            a1s = safe_float(vals[1]) if len(vals) > 1 else None
            a2s = safe_float(vals[2]) if len(vals) > 2 else None
        else:
            a0 = a1s = a2s = None
        chart_data_rows.append((seed, a0, a1s, a2s))

        c = ws3.cell(row=row, column=1, value=seed)
        sc(c, size=9, fill=FILL_LIGHT_BLUE, border=thin_border(), halign="center")
        for col, val, ap_fill in [(2, a0, FILL_A0), (3, a1s, FILL_A1S), (4, a2s, FILL_A2S)]:
            c = ws3.cell(row=row, column=col, value=val)
            sc(c, size=9, fill=ap_fill, border=thin_border(),
               num_fmt="#,##0.000", halign="center")

    n_data_rows = len(RUNS)
    last_data_row = data_start_row + n_data_rows

    # ---- Line chart: Total cost per seed for all 3 approaches ----
    lc = LineChart()
    lc.title    = "Total Cost per Seed — All Three Approaches"
    lc.style    = 10
    lc.y_axis.title = "Total Cost (USD)"
    lc.x_axis.title = "Seed"
    lc.width    = 24
    lc.height   = 14
    lc.grouping = "standard"
    lc.smooth   = False

    for col_idx, ap_name, color in [
        (2, "A0 (One-Shot)",  "4472C4"),
        (3, "A1S (CE MPC)",   "ED7D31"),
        (4, "A2S (SB MPC)",   "70AD47"),
    ]:
        data_ref = Reference(ws3,
                             min_col=col_idx, max_col=col_idx,
                             min_row=data_start_row,
                             max_row=last_data_row)
        lc.add_data(data_ref, titles_from_data=True)
        lc.series[-1].graphicalProperties.line.solidFill = color
        lc.series[-1].graphicalProperties.line.width = 15000  # 1.5 pt in EMUs

    cats_ref = Reference(ws3, min_col=1, max_col=1,
                         min_row=data_start_row + 1, max_row=last_data_row)
    lc.set_categories(cats_ref)
    ws3.add_chart(lc, "F3")

    # ---- Savings vs A0 line chart (cols E-F) ----
    hdr(ws3.cell(row=data_start_row, column=5), "A1S Saving vs A0", level="sub")
    hdr(ws3.cell(row=data_start_row, column=6), "A2S Saving vs A0", level="sub")

    for i, (seed, a0, a1s, a2s) in enumerate(chart_data_rows):
        row = data_start_row + 1 + i
        d1 = (a0 - a1s) if (a0 and a1s) else None
        d2 = (a0 - a2s) if (a0 and a2s) else None
        c = ws3.cell(row=row, column=5, value=d1)
        sc(c, size=9, fill=FILL_A1S, border=thin_border(),
           num_fmt="+#,##0.000;-#,##0.000;0.000", halign="center")
        c = ws3.cell(row=row, column=6, value=d2)
        sc(c, size=9, fill=FILL_A2S, border=thin_border(),
           num_fmt="+#,##0.000;-#,##0.000;0.000", halign="center")

    # Savings line chart
    sc2 = LineChart()
    sc2.title    = "Cost Saving vs A0 per Seed (positive = better than A0)"
    sc2.style    = 10
    sc2.y_axis.title = "Saving (USD)"
    sc2.x_axis.title = "Seed"
    sc2.width    = 24
    sc2.height   = 14
    sc2.grouping = "standard"

    for col_idx, ap_name, color in [
        (5, "A1S Saving vs A0", "ED7D31"),
        (6, "A2S Saving vs A0", "70AD47"),
    ]:
        data_ref = Reference(ws3,
                             min_col=col_idx, max_col=col_idx,
                             min_row=data_start_row,
                             max_row=last_data_row)
        sc2.add_data(data_ref, titles_from_data=True)
        sc2.series[-1].graphicalProperties.line.solidFill = color

    sc2.set_categories(cats_ref)
    ws3.add_chart(sc2, "F33")

    # ---- Bar chart: Win counts ----
    bar_data_row = last_data_row + 3
    ws3.merge_cells(start_row=bar_data_row - 1, start_column=1,
                    end_row=bar_data_row - 1, end_column=4)
    c = ws3.cell(row=bar_data_row - 1, column=1, value="Win Count Data (for chart)")
    sc(c, bold=True, size=10, color=C_HEADER_TXT,
       fill=FILL_MED_BLUE, halign="left", border=thin_border())

    hdr(ws3.cell(row=bar_data_row, column=1), "Approach", level="sub", halign="left")
    hdr(ws3.cell(row=bar_data_row, column=2), "Wins",      level="sub")

    bar_ap_rows = [
        ("A0 (One-Shot)", win_counts["A0"],  FILL_A0),
        ("A1S (CE MPC)",  win_counts["A1S"], FILL_A1S),
        ("A2S (SB MPC)",  win_counts["A2S"], FILL_A2S),
        ("Tie",           win_counts["Tie"], FILL_LIGHT_BLUE),
    ]
    for r_off, (name, cnt, ap_fill) in enumerate(bar_ap_rows, 1):
        row = bar_data_row + r_off
        c = ws3.cell(row=row, column=1, value=name)
        sc(c, size=10, fill=ap_fill, border=thin_border(), halign="left")
        c = ws3.cell(row=row, column=2, value=cnt)
        sc(c, bold=True, size=10, fill=ap_fill, border=thin_border(),
           num_fmt="0", halign="center")

    bc = BarChart()
    bc.type    = "col"
    bc.style   = 10
    bc.title   = "Number of Runs Won by Each Approach"
    bc.y_axis.title = "Number of Wins"
    bc.x_axis.title = "Approach"
    bc.width   = 14
    bc.height  = 12

    bar_vals = Reference(ws3, min_col=2, max_col=2,
                         min_row=bar_data_row,
                         max_row=bar_data_row + len(bar_ap_rows))
    bar_cats = Reference(ws3, min_col=1, max_col=1,
                         min_row=bar_data_row + 1,
                         max_row=bar_data_row + len(bar_ap_rows))
    bc.add_data(bar_vals, titles_from_data=True)
    bc.set_categories(bar_cats)
    bc.series[0].graphicalProperties.solidFill = "2E75B6"
    ws3.add_chart(bc, "F63")

    # ---- Summary stats mini-table for box-plot style chart ----
    box_row = bar_data_row + len(bar_ap_rows) + 3
    ws3.merge_cells(start_row=box_row - 1, start_column=1,
                    end_row=box_row - 1, end_column=5)
    c = ws3.cell(row=box_row - 1, column=1,
                 value="Distribution Summary (for box-plot style chart)")
    sc(c, bold=True, size=10, color=C_HEADER_TXT,
       fill=FILL_MED_BLUE, halign="left", border=thin_border())

    box_hdrs = ["Approach", "Min", "Q1", "Median", "Max"]
    for col, h in enumerate(box_hdrs, 1):
        c = ws3.cell(row=box_row, column=col, value=h)
        sc(c, bold=True, size=10, color=C_HEADER_TXT,
           fill=FILL_MED_BLUE, border=thin_border())

    box_data = [
        ("A0", list(a0_vals),  FILL_A0),
        ("A1S", list(a1s_vals), FILL_A1S),
        ("A2S", list(a2s_vals), FILL_A2S),
    ]
    box_chart_data = {}
    for r_off, (name, vals, ap_fill) in enumerate(box_data, 1):
        sv = stats(vals)
        # sv = [count, mean, median, std, min, q1, max]
        count, s_mean, s_median, s_std, s_min, s_q1, s_max = sv
        row = box_row + r_off
        box_chart_data[name] = (s_min, s_q1, s_median, s_max)
        c = ws3.cell(row=row, column=1, value=name)
        sc(c, bold=True, size=10, fill=ap_fill, border=thin_border(), halign="left")
        for col, val in [(2, s_min), (3, s_q1), (4, s_median), (5, s_max)]:
            c = ws3.cell(row=row, column=col, value=val)
            sc(c, size=10, fill=ap_fill, border=thin_border(),
               num_fmt="#,##0.000", halign="center")

    # Bar chart approximating a box plot (stacked: min, Q1-min, median-Q1, max-median)
    # Using a line chart of min/median/max per approach for simplicity
    range_chart = LineChart()
    range_chart.title  = "Cost Range per Approach (Min / Median / Max)"
    range_chart.style  = 10
    range_chart.y_axis.title = "Total Cost (USD)"
    range_chart.x_axis.title = "Approach"
    range_chart.width  = 14
    range_chart.height = 12

    for col_idx, label, color in [
        (2, "Min",    "70AD47"),
        (4, "Median", "ED7D31"),
        (5, "Max",    "4472C4"),
    ]:
        ref = Reference(ws3, min_col=col_idx, max_col=col_idx,
                        min_row=box_row, max_row=box_row + len(box_data))
        range_chart.add_data(ref, titles_from_data=True)
        range_chart.series[-1].graphicalProperties.line.solidFill = color

    cats_box = Reference(ws3, min_col=1, max_col=1,
                         min_row=box_row + 1, max_row=box_row + len(box_data))
    range_chart.set_categories(cats_box)
    ws3.add_chart(range_chart, "F78")

    ws3.column_dimensions["A"].width = 16
    ws3.column_dimensions["B"].width = 16
    ws3.column_dimensions["C"].width = 16
    ws3.column_dimensions["D"].width = 16
    ws3.column_dimensions["E"].width = 16

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    wb.save(out_path)
    print(f"  [OK] Cost Analysis saved: {out_path}")


# ###########################################################################
# ###########################################################################
#   PART B — PLOTS  (all data comes from the Excel files written in Part A)
# ###########################################################################
# ###########################################################################

# ---------------------------------------------------------------------------
# Colour palette (consistent across all figures)
# ---------------------------------------------------------------------------
C = {
    "A0":  "#4472C4",   # blue
    "A1S": "#ED7D31",   # orange
    "A2S": "#70AD47",   # green
    "bg":  "#F8F9FA",
    "grid":"#E0E0E0",
    "text":"#1F1F1F",
}
AP_COLOURS = [C["A0"], C["A1S"], C["A2S"]]
AP_NAMES   = ["A0 (One-Shot)", "A1S (CE MPC)", "A2S (SB MPC)"]
AP_KEYS    = ["A0", "A1S", "A2S"]

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.facecolor":   C["bg"],
    "axes.edgecolor":   "#CCCCCC",
    "axes.grid":        True,
    "grid.color":       C["grid"],
    "grid.linestyle":   "--",
    "grid.alpha":       0.7,
    "figure.facecolor": "white",
    "figure.dpi":       150,
    "savefig.dpi":      150,
    "savefig.bbox":     "tight",
    "savefig.facecolor":"white",
    "legend.framealpha":0.9,
    "legend.edgecolor": "#CCCCCC",
})


def valid_pairs(*arrays):
    """Return only indices where all arrays have non-None values."""
    idx = [i for i in range(len(arrays[0]))
           if all(a[i] is not None for a in arrays)]
    return idx


# ---------------------------------------------------------------------------
# Helper: add a neat title bar to each figure
# ---------------------------------------------------------------------------
def fig_title(fig, text, subtitle=""):
    fig.suptitle(text, fontsize=14, fontweight="bold", color=C["text"], y=0.98)
    if subtitle:
        fig.text(0.5, 0.955, subtitle, ha="center", fontsize=10,
                 color="#555555", style="italic")


def save(fig, path: Path, name: str):
    out = path / name
    fig.savefig(out)
    plt.close(fig)
    print(f"  [saved] {out.name}")
    return out


# ===========================================================================
# FIGURE 1 — Line chart: total cost per seed, all 3 approaches
# ===========================================================================
def fig_total_cost_line(seeds, a0v, a1sv, a2sv, out_dir):
    idx = valid_pairs(a0v, a1sv, a2sv)
    s   = [seeds[i] for i in idx]
    v0  = [a0v[i]   for i in idx]
    v1  = [a1sv[i]  for i in idx]
    v2  = [a2sv[i]  for i in idx]

    fig, ax = plt.subplots(figsize=(18, 6))
    ax.plot(s, v0, color=C["A0"],  lw=1.4, alpha=0.85, label=AP_NAMES[0], zorder=3)
    ax.plot(s, v1, color=C["A1S"], lw=1.4, alpha=0.85, label=AP_NAMES[1], zorder=3)
    ax.plot(s, v2, color=C["A2S"], lw=1.4, alpha=0.85, label=AP_NAMES[2], zorder=3)

    # Mean lines
    for v, col, key in zip([v0, v1, v2], AP_COLOURS, AP_KEYS):
        ax.axhline(mean(v), color=col, lw=1.2, ls="--", alpha=0.55)

    ax.set_xlabel("Seed", fontsize=12)
    ax.set_ylabel("Total Cost (USD)", fontsize=12)
    ax.legend(fontsize=10, loc="upper right")
    ax.set_xlim(min(s), max(s))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))

    # Annotate means in margin
    for v, col, name in zip([v0, v1, v2], AP_COLOURS, AP_NAMES):
        ax.annotate(f"μ={mean(v):.2f}", xy=(max(s) + 0.5, mean(v)),
                    fontsize=8, color=col, va="center")

    fig_title(fig, "Total Cost per Seed — All Three Approaches",
              subtitle="Dashed lines show per-approach mean")
    return save(fig, out_dir, "01_total_cost_per_seed.png")


# ===========================================================================
# FIGURE 2 — Savings vs A0 per seed
# ===========================================================================
def fig_savings_vs_a0(seeds, a0v, a1sv, a2sv, out_dir):
    idx = valid_pairs(a0v, a1sv, a2sv)
    s   = [seeds[i] for i in idx]
    d1  = [a0v[i] - a1sv[i] for i in idx]   # positive = A1S saves vs A0
    d2  = [a0v[i] - a2sv[i] for i in idx]   # positive = A2S saves vs A0

    fig, axes = plt.subplots(2, 1, figsize=(18, 9), sharex=True)

    for ax, diffs, col, name, key in zip(
            axes,
            [d1, d2],
            [C["A1S"], C["A2S"]],
            ["A1S (CE MPC) saving vs A0", "A2S (SB MPC) saving vs A0"],
            ["A1S", "A2S"]):
        colours = [col if v >= 0 else "#E74C3C" for v in diffs]
        ax.bar(s, diffs, color=colours, width=0.7, zorder=3)
        ax.axhline(0,        color="#555555", lw=0.8, zorder=4)
        ax.axhline(mean(diffs), color=col, lw=1.5, ls="--", alpha=0.7,
                   label=f"Mean saving = ${mean(diffs):.3f}")
        ax.set_ylabel("Saving (USD)", fontsize=11)
        ax.set_title(name, fontsize=11, fontweight="bold", color="#333333")
        ax.legend(fontsize=10)
        # Shade negative region
        ax.fill_between(s, 0, [min(0, v) for v in diffs],
                        alpha=0.08, color="#E74C3C", zorder=2)

    axes[-1].set_xlabel("Seed", fontsize=12)
    axes[-1].xaxis.set_major_locator(ticker.MultipleLocator(5))

    fig_title(fig, "Cost Saving vs A0 per Seed",
              subtitle="Green bars = approach beats A0  |  Red bars = approach worse than A0")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return save(fig, out_dir, "02_savings_vs_A0.png")


# ===========================================================================
# FIGURE 3 — Box plots of total cost distribution
# ===========================================================================
def fig_boxplots(a0v, a1sv, a2sv, out_dir):
    idx  = valid_pairs(a0v, a1sv, a2sv)
    data = [[a0v[i] for i in idx],
            [a1sv[i] for i in idx],
            [a2sv[i] for i in idx]]

    fig, ax = plt.subplots(figsize=(9, 7))
    bp = ax.boxplot(data, patch_artist=True, widths=0.45,
                    medianprops=dict(color="#333333", lw=2),
                    whiskerprops=dict(lw=1.2),
                    capprops=dict(lw=1.5),
                    flierprops=dict(marker="o", markersize=5, alpha=0.5))

    for patch, col in zip(bp["boxes"], AP_COLOURS):
        patch.set_facecolor(col)
        patch.set_alpha(0.65)

    # Overlay individual points (strip plot)
    for j, (vals, col) in enumerate(zip(data, AP_COLOURS), 1):
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
        ax.scatter(np.full(len(vals), j) + jitter, vals,
                   color=col, alpha=0.35, s=18, zorder=3)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(AP_NAMES, fontsize=11)
    ax.set_ylabel("Total Cost (USD)", fontsize=12)

    # Annotate median
    for j, vals in enumerate(data, 1):
        med = sorted(vals)[len(vals) // 2]
        ax.text(j, med, f" {med:.2f}", va="center", fontsize=8,
                color="#333333", fontweight="bold")

    fig_title(fig, "Total Cost Distribution per Approach",
              subtitle="Box = IQR  |  Whiskers = 1.5×IQR  |  Dots = individual seeds")
    return save(fig, out_dir, "03_boxplots.png")


# ===========================================================================
# FIGURE 4 — Win count bar chart + pie chart
# ===========================================================================
def fig_win_counts(seeds, a0v, a1sv, a2sv, out_dir):
    idx = valid_pairs(a0v, a1sv, a2sv)
    wins = {"A0": 0, "A1S": 0, "A2S": 0, "Tie": 0}
    for i in idx:
        best = min(a0v[i], a1sv[i], a2sv[i])
        w = [k for k, v in [("A0", a0v[i]), ("A1S", a1sv[i]), ("A2S", a2sv[i])]
             if abs(v - best) < 1e-6]
        if len(w) > 1: wins["Tie"] += 1
        else:          wins[w[0]] += 1

    fig, (ax_bar, ax_pie) = plt.subplots(1, 2, figsize=(13, 6))

    labels = ["A0\n(One-Shot)", "A1S\n(CE MPC)", "A2S\n(SB MPC)", "Tie"]
    counts = [wins["A0"], wins["A1S"], wins["A2S"], wins["Tie"]]
    colours_bar = [C["A0"], C["A1S"], C["A2S"], "#AAAAAA"]

    bars = ax_bar.bar(labels, counts, color=colours_bar, width=0.55,
                      edgecolor="white", linewidth=1.2, zorder=3)
    for bar, cnt in zip(bars, counts):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.4,
                    str(cnt), ha="center", va="bottom",
                    fontsize=13, fontweight="bold")
    ax_bar.set_ylabel("Number of Seeds Won", fontsize=12)
    ax_bar.set_ylim(0, max(counts) * 1.2)
    ax_bar.set_title("Win Counts (lowest total cost wins)", fontsize=12,
                     fontweight="bold")

    # Pie (exclude ties for clarity if zero)
    pie_labels = ["A0", "A1S", "A2S"]
    pie_vals   = [wins["A0"], wins["A1S"], wins["A2S"]]
    pie_cols   = [C["A0"], C["A1S"], C["A2S"]]
    if wins["Tie"] > 0:
        pie_labels.append("Tie")
        pie_vals.append(wins["Tie"])
        pie_cols.append("#AAAAAA")

    wedges, texts, autotexts = ax_pie.pie(
        pie_vals, labels=pie_labels, colors=pie_cols,
        autopct=lambda p: f"{p:.1f}%\n({int(round(p*sum(pie_vals)/100))})",
        startangle=140, pctdistance=0.75,
        wedgeprops=dict(linewidth=1.2, edgecolor="white"))
    for t in autotexts:
        t.set_fontsize(10)
        t.set_fontweight("bold")
    ax_pie.set_title("Win Share", fontsize=12, fontweight="bold")

    fig_title(fig, "Approach Win Counts across 100 Seeds")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return save(fig, out_dir, "04_win_counts.png")


# ===========================================================================
# FIGURE 5 — Chain analysis: how often does A0 ≥ A1S ≥ A2S hold?
# ===========================================================================
def fig_chain_analysis(seeds, a0v, a1sv, a2sv, out_dir):
    idx = valid_pairs(a0v, a1sv, a2sv)
    n   = len(idx)

    strict   = sum(1 for i in idx if a2sv[i] < a1sv[i] < a0v[i])
    nstrict  = sum(1 for i in idx if a0v[i] >= a1sv[i] >= a2sv[i])
    broken   = n - nstrict

    # Per-seed chain status for the scatter
    s_ok, c_ok     = [], []
    s_brk, c_brk   = [], []
    for i in idx:
        a0, a1, a2 = a0v[i], a1sv[i], a2sv[i]
        if a0 >= a1 >= a2:
            s_ok.append(seeds[i]);  c_ok.append(a2)   # lowest cost (A2S) when chain holds
        else:
            s_brk.append(seeds[i]); c_brk.append(min(a0, a1, a2))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: stacked bar showing chain holds vs broken
    categories = ["Strict\nA2S<A1S<A0", "Non-strict\nA0≥A1S≥A2S", "Chain\nBroken"]
    counts_c   = [strict, nstrict, broken]
    bar_cols   = ["#2ECC71", "#58D68D", "#E74C3C"]
    bars = axes[0].bar(categories, counts_c, color=bar_cols,
                       width=0.5, edgecolor="white", lw=1.2, zorder=3)
    for bar, cnt in zip(bars, counts_c):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.4,
                     f"{cnt}\n({cnt/n*100:.1f}%)",
                     ha="center", va="bottom", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Number of Seeds", fontsize=12)
    axes[0].set_ylim(0, n * 1.15)
    axes[0].set_title("Chain A0 ≥ A1S ≥ A2S Analysis", fontsize=12,
                       fontweight="bold")

    # Right: scatter seed vs best cost, coloured by chain status
    axes[1].scatter(s_ok,  c_ok,  color="#2ECC71", alpha=0.7, s=40,
                    label=f"Chain holds ({len(s_ok)} seeds)", zorder=3)
    axes[1].scatter(s_brk, c_brk, color="#E74C3C", alpha=0.7, s=40,
                    marker="x", linewidths=1.5,
                    label=f"Chain broken ({len(s_brk)} seeds)", zorder=3)
    axes[1].set_xlabel("Seed", fontsize=12)
    axes[1].set_ylabel("Best (lowest) Total Cost (USD)", fontsize=12)
    axes[1].set_title("Chain Status per Seed", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=10)
    axes[1].xaxis.set_major_locator(ticker.MultipleLocator(10))

    fig_title(fig, "Ordering Chain Analysis: A0 ≥ A1S ≥ A2S",
              subtitle="Does the expected A0 > A1S > A2S cost ranking hold in practice?")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return save(fig, out_dir, "05_chain_analysis.png")


# ===========================================================================
# FIGURE 6 — Cumulative distribution function (CDF) of total cost
# ===========================================================================
def fig_cdf(a0v, a1sv, a2sv, out_dir):
    idx = valid_pairs(a0v, a1sv, a2sv)

    fig, ax = plt.subplots(figsize=(10, 7))
    for vals, col, name in zip(
            [[a0v[i] for i in idx],
             [a1sv[i] for i in idx],
             [a2sv[i] for i in idx]],
            AP_COLOURS, AP_NAMES):
        sv = sorted(vals)
        y  = np.arange(1, len(sv) + 1) / len(sv)
        ax.step(sv, y, color=col, lw=2, label=name, where="post")
        ax.axvline(mean(sv), color=col, lw=1.2, ls="--", alpha=0.5)

    ax.set_xlabel("Total Cost (USD)", fontsize=12)
    ax.set_ylabel("Cumulative Probability", fontsize=12)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.05)

    fig_title(fig, "Cumulative Distribution of Total Cost",
              subtitle="Dashed verticals = per-approach mean  |  "
                        "Leftmost curve = cheapest approach most often")
    return save(fig, out_dir, "06_cost_cdf.png")


# ===========================================================================
# FIGURE 7 — Pairwise scatter: A1S vs A0, A2S vs A0, A2S vs A1S
# ===========================================================================
def fig_pairwise_scatter(a0v, a1sv, a2sv, out_dir):
    idx = valid_pairs(a0v, a1sv, a2sv)
    v0  = [a0v[i]   for i in idx]
    v1  = [a1sv[i]  for i in idx]
    v2  = [a2sv[i]  for i in idx]

    pairs = [
        (v0,  v1,  "A0 Total Cost (USD)", "A1S Total Cost (USD)",
         C["A1S"], "A1S vs A0"),
        (v0,  v2,  "A0 Total Cost (USD)", "A2S Total Cost (USD)",
         C["A2S"], "A2S vs A0"),
        (v1,  v2,  "A1S Total Cost (USD)", "A2S Total Cost (USD)",
         C["A0"],  "A2S vs A1S"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (x, y, xl, yl, col, title) in zip(axes, pairs):
        ax.scatter(x, y, color=col, alpha=0.55, s=30, zorder=3)
        lo = min(min(x), min(y))
        hi = max(max(x), max(y))
        ax.plot([lo, hi], [lo, hi], color="#555555", lw=1.2, ls="--",
                label="y = x (tie line)", zorder=4)
        # Count above / below the tie line
        above = sum(1 for xi, yi in zip(x, y) if yi > xi)  # y approach costs more
        below = sum(1 for xi, yi in zip(x, y) if yi < xi)  # y approach costs less
        ax.text(0.03, 0.96, f"← {below} seeds: Y cheaper",
                transform=ax.transAxes, fontsize=9, va="top", color="#27AE60")
        ax.text(0.97, 0.04, f"{above} seeds: Y more expensive →",
                transform=ax.transAxes, fontsize=9, ha="right", color="#E74C3C")
        ax.set_xlabel(xl, fontsize=10)
        ax.set_ylabel(yl, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)

    fig_title(fig, "Pairwise Total Cost Scatter",
              subtitle="Points below the dashed line: Y-axis approach is cheaper than X-axis approach")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return save(fig, out_dir, "07_pairwise_scatter.png")


# ===========================================================================
# FIGURE 8 — Per-metric mean comparison (grouped bar)
# ===========================================================================
def fig_per_metric_means(all_metrics, out_dir):
    # Skip metrics that are uniformly zero or near-zero across all approaches
    metrics_to_plot = []
    for m in EXPECTED_METRICS:
        vals_a0  = [v for v in all_metrics[m]["A0"]  if v is not None]
        vals_a1s = [v for v in all_metrics[m]["A1S"] if v is not None]
        vals_a2s = [v for v in all_metrics[m]["A2S"] if v is not None]
        if not vals_a0: continue
        if max(abs(mean(vals_a0)), abs(mean(vals_a1s)), abs(mean(vals_a2s))) < 1e-6:
            continue
        metrics_to_plot.append(m)

    n  = len(metrics_to_plot)
    x  = np.arange(n)
    w  = 0.26

    fig, ax = plt.subplots(figsize=(max(14, n * 1.2), 7))
    for j, (key, col, name) in enumerate(
            zip(AP_KEYS, AP_COLOURS, AP_NAMES)):
        means = []
        for m in metrics_to_plot:
            vals = [v for v in all_metrics[m][key] if v is not None]
            means.append(mean(vals) if vals else 0)
        ax.bar(x + (j - 1) * w, means, w, label=name,
               color=col, alpha=0.8, edgecolor="white", lw=0.8, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Mean Value (units vary per metric)", fontsize=11)
    ax.legend(fontsize=10)

    fig_title(fig, "Mean KPI Value per Metric — All Three Approaches",
              subtitle="Averaged across all 100 seeds")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return save(fig, out_dir, "08_per_metric_means.png")


# ===========================================================================
# FIGURE 9 — Rolling mean of total cost (window=10) to show trends
# ===========================================================================
def fig_rolling_mean(seeds, a0v, a1sv, a2sv, out_dir):
    idx = valid_pairs(a0v, a1sv, a2sv)
    s   = [seeds[i] for i in idx]
    window = 10

    def rolling(vals, w):
        return [mean(vals[max(0, i - w + 1):i + 1]) for i in range(len(vals))]

    fig, ax = plt.subplots(figsize=(14, 6))
    for v, col, name in zip(
            [[a0v[i] for i in idx],
             [a1sv[i] for i in idx],
             [a2sv[i] for i in idx]],
            AP_COLOURS, AP_NAMES):
        ax.plot(s, v, color=col, lw=0.6, alpha=0.25, zorder=2)
        rv = rolling(v, window)
        ax.plot(s, rv, color=col, lw=2.2, label=f"{name} (rolling {window}-seed avg)",
                zorder=3)

    ax.set_xlabel("Seed", fontsize=12)
    ax.set_ylabel("Total Cost (USD)", fontsize=12)
    ax.legend(fontsize=10)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))

    fig_title(fig, f"Rolling {window}-Seed Mean of Total Cost",
              subtitle="Faint lines = raw per-seed cost  |  Solid = rolling average")
    return save(fig, out_dir, "09_rolling_mean.png")


# ===========================================================================
# FIGURE 10 — Histogram of total cost per approach (overlaid)
# ===========================================================================
def fig_histogram(a0v, a1sv, a2sv, out_dir):
    idx = valid_pairs(a0v, a1sv, a2sv)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    all_vals = [[a0v[i] for i in idx],
                [a1sv[i] for i in idx],
                [a2sv[i] for i in idx]]

    global_min = min(min(v) for v in all_vals)
    global_max = max(max(v) for v in all_vals)
    bins = np.linspace(global_min, global_max, 22)

    for ax, vals, col, name in zip(axes, all_vals, AP_COLOURS, AP_NAMES):
        ax.hist(vals, bins=bins, color=col, alpha=0.75,
                edgecolor="white", lw=0.8, zorder=3)
        ax.axvline(mean(vals), color="#333333", lw=1.8, ls="--",
                   label=f"Mean = {mean(vals):.2f}")
        ax.axvline(median(vals), color="#333333", lw=1.2, ls=":",
                   label=f"Median = {median(vals):.2f}")
        ax.set_title(name, fontsize=11, fontweight="bold", color="#333333")
        ax.set_xlabel("Total Cost (USD)", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.legend(fontsize=9)

    fig_title(fig, "Total Cost Distribution — Histogram per Approach",
              subtitle="Dashed = mean  |  Dotted = median")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return save(fig, out_dir, "10_histograms.png")


# ===========================================================================
# FIGURE 11 — Heatmap of all KPIs (normalised)
# ===========================================================================
def fig_kpi_heatmap(all_metrics, out_dir):
    try:
        import matplotlib.colors as mcolors
    except ImportError:
        print("  [SKIP] heatmap — matplotlib.colors not available")
        return

    metrics_to_plot = []
    matrix = []   # shape: (n_metrics, 3)

    for m in EXPECTED_METRICS:
        row = []
        for key in AP_KEYS:
            vals = [v for v in all_metrics[m][key] if v is not None]
            row.append(mean(vals) if vals else 0.0)
        # Skip rows that are all zero
        if max(abs(v) for v in row) < 1e-6:
            continue
        metrics_to_plot.append(m)
        # Normalise row to [0, 1] within metric (so colour = relative rank)
        rmin, rmax = min(row), max(row)
        rng = rmax - rmin if rmax != rmin else 1.0
        matrix.append([(v - rmin) / rng for v in row])

    mat = np.array(matrix)   # (n_metrics, 3)

    fig, ax = plt.subplots(figsize=(7, max(6, len(metrics_to_plot) * 0.55)))
    im = ax.imshow(mat, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(AP_NAMES, fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(metrics_to_plot)))
    ax.set_yticklabels(metrics_to_plot, fontsize=9)

    # Annotate with raw mean values
    all_raw = []
    for m in metrics_to_plot:
        for key in AP_KEYS:
            vals = [v for v in all_metrics[m][key] if v is not None]
            all_raw.append(mean(vals) if vals else 0.0)

    k = 0
    for r in range(len(metrics_to_plot)):
        for c in range(3):
            val = all_raw[k]; k += 1
            ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                    fontsize=7.5, color="black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Relative value within metric\n(green = lower/better, red = higher/worse)",
                   fontsize=9)

    fig_title(fig, "KPI Heatmap — Mean Values across 100 Seeds",
              subtitle="Colour scale is per-row (per metric); values shown are actual means")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return save(fig, out_dir, "11_kpi_heatmap.png")


# ===========================================================================
# FIGURE 12 — Summary card (one-page overview)
# ===========================================================================
def fig_summary_card(seeds, a0v, a1sv, a2sv, out_dir):
    idx = valid_pairs(a0v, a1sv, a2sv)
    n   = len(idx)
    v   = {k: [d[i] for i in idx]
           for k, d in [("A0", a0v), ("A1S", a1sv), ("A2S", a2sv)]}

    wins = {"A0": 0, "A1S": 0, "A2S": 0, "Tie": 0}
    chain_ok = 0
    for i in idx:
        best_val = min(a0v[i], a1sv[i], a2sv[i])
        w = [k for k, d in [("A0", a0v), ("A1S", a1sv), ("A2S", a2sv)]
             if abs(d[i] - best_val) < 1e-6]
        wins["Tie" if len(w) > 1 else w[0]] += 1
        if a0v[i] >= a1sv[i] >= a2sv[i]: chain_ok += 1

    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor("white")

    # Title banner
    fig.text(0.5, 0.96, "Ground Truth 100 Runs — Summary Overview",
             ha="center", va="top", fontsize=16, fontweight="bold", color="#1F4E79")
    fig.text(0.5, 0.93,
             f"All {n} seeds completed  |  mode = live_data  |  1200s solver limit",
             ha="center", va="top", fontsize=10, color="#555555", style="italic")

    gs = fig.add_gridspec(2, 3, left=0.07, right=0.97,
                          top=0.88, bottom=0.07, hspace=0.40, wspace=0.35)

    # ----- Top-left: mean cost bar -----
    ax1 = fig.add_subplot(gs[0, 0])
    means = [mean(v[k]) for k in AP_KEYS]
    bars  = ax1.bar(AP_NAMES, means, color=AP_COLOURS, width=0.5,
                    edgecolor="white", lw=1.2, zorder=3)
    for bar, m in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.1,
                 f"${m:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.set_title("Mean Total Cost", fontsize=11, fontweight="bold")
    ax1.set_ylabel("USD")
    ax1.set_ylim(0, max(means) * 1.18)
    plt.setp(ax1.get_xticklabels(), fontsize=8)

    # ----- Top-middle: win count -----
    ax2 = fig.add_subplot(gs[0, 1])
    wlabels = ["A0", "A1S", "A2S", "Tie"]
    wcounts = [wins[k] for k in wlabels]
    wcolors = [C["A0"], C["A1S"], C["A2S"], "#AAAAAA"]
    ax2.bar(wlabels, wcounts, color=wcolors, width=0.5,
            edgecolor="white", lw=1.2, zorder=3)
    for x, cnt in enumerate(wcounts):
        ax2.text(x, cnt + 0.3, str(cnt),
                 ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax2.set_title("Wins (lowest cost)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Seeds")
    ax2.set_ylim(0, max(wcounts) * 1.2)

    # ----- Top-right: chain holds -----
    ax3 = fig.add_subplot(gs[0, 2])
    ch_labels  = ["Chain\nholds", "Chain\nbroken"]
    ch_counts  = [chain_ok, n - chain_ok]
    ch_colors  = ["#2ECC71", "#E74C3C"]
    ax3.bar(ch_labels, ch_counts, color=ch_colors, width=0.45,
            edgecolor="white", lw=1.2, zorder=3)
    for x, (cnt, tot) in enumerate(zip(ch_counts, [n, n])):
        ax3.text(x, cnt + 0.3, f"{cnt}\n({cnt/n*100:.0f}%)",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax3.set_title("A0 ≥ A1S ≥ A2S chain", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Seeds")
    ax3.set_ylim(0, n * 1.25)

    # ----- Bottom row: cost line chart -----
    ax4 = fig.add_subplot(gs[1, :])
    s_plot = [seeds[i] for i in idx]
    for vals, col, name in zip(
            [v["A0"], v["A1S"], v["A2S"]], AP_COLOURS, AP_NAMES):
        ax4.plot(s_plot, vals, color=col, lw=1.1, alpha=0.8, label=name)
        ax4.axhline(mean(vals), color=col, lw=1.0, ls="--", alpha=0.45)
    ax4.set_xlabel("Seed", fontsize=11)
    ax4.set_ylabel("Total Cost (USD)", fontsize=11)
    ax4.legend(fontsize=9, loc="upper right")
    ax4.set_title("Total Cost per Seed", fontsize=11, fontweight="bold")
    ax4.xaxis.set_major_locator(ticker.MultipleLocator(5))

    return save(fig, out_dir, "12_summary_overview.png")


# ===========================================================================
# Read the generated Excel files back in (this is the data source for plots)
# ===========================================================================
def load_totals_from_excel(xl_path: Path):
    """Reads the 'Total Cost' sheet -> seeds, a0_vals, a1s_vals, a2s_vals
    (None where a run is missing)."""
    wb = openpyxl.load_workbook(xl_path, data_only=True)
    ws = wb["Total Cost"]
    seeds, a0v, a1sv, a2sv = [], [], [], []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[0] is None:
            continue
        seeds.append(row[0])
        a0v.append(safe_float(row[2]))     # "MISSING" / blanks -> None
        a1sv.append(safe_float(row[3]))
        a2sv.append(safe_float(row[4]))
    wb.close()
    return seeds, a0v, a1sv, a2sv


def load_metrics_from_excel(xl_path: Path):
    """Reads the 'KPI Matrix' sheet -> {metric: {"A0": [...], "A1S": [...], "A2S": [...]}}
    with one entry per run (None where a run is missing)."""
    wb = openpyxl.load_workbook(xl_path, data_only=True)
    ws = wb["KPI Matrix"]
    n_ap = len(APPROACH_SHORT)
    all_metrics = {m: {k: [] for k in APPROACH_SHORT} for m in EXPECTED_METRICS}

    for row in ws.iter_rows(min_row=3, values_only=True):
        metric = row[0]
        if metric not in all_metrics:
            continue
        for i in range(len(RUNS)):
            col_s = 1 + i * n_ap            # 0-based index of column B + run offset
            for j, key in enumerate(APPROACH_SHORT):
                idx = col_s + j
                all_metrics[metric][key].append(
                    safe_float(row[idx]) if idx < len(row) else None)
    wb.close()
    return all_metrics


# ===========================================================================
# Main
# ===========================================================================
def main():
    default_root = Path(
        r"C:\Users\shubh\Desktop\MPC\Actual Run Data\Ground Truth 100 runs data"
        r"\Output_GroundTruth_100Runs"
    )
    default_out_dir = Path(
        r"C:\Users\shubh\Desktop\MPC\Actual Run Data\Ground Truth 100 runs data"
    )

    root    = Path(sys.argv[1]) if len(sys.argv) > 1 else default_root
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else default_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Source root : {root}")
    print(f"Output dir  : {out_dir}")
    print(f"Runs        : {len(RUNS)}")
    print("-" * 60)

    if not root.exists():
        print(f"[WARN] Root directory not found: {root}")
        print("       Running in demo mode — all runs will show as missing.")

    # ------------------------------------------------------------------
    # STEP 1 — HTML  ->  Excel
    # ------------------------------------------------------------------
    print("STEP 1: compiling Excel workbooks from HTML ...")
    run_data = load_all(root)
    present  = sum(1 for d in run_data if d is not None)
    print(f"  Parsed {present} / {len(RUNS)} runs successfully.")
    print()

    kpi_path  = out_dir / "GroundTruth_100Runs_KPI_Matrix.xlsx"
    cost_path = out_dir / "GroundTruth_100Runs_TotalCost_Analysis.xlsx"

    build_kpi_matrix(root, kpi_path, run_data)
    build_cost_analysis(root, cost_path, run_data)

    # ------------------------------------------------------------------
    # STEP 2 — Excel  ->  plots
    # ------------------------------------------------------------------
    print()
    print("STEP 2: reading the generated Excel files for plotting ...")
    seeds, a0v, a1sv, a2sv = load_totals_from_excel(cost_path)
    all_metrics            = load_metrics_from_excel(kpi_path)
    n_ok = len(valid_pairs(a0v, a1sv, a2sv))
    print(f"  Read {len(seeds)} rows from Excel ({n_ok} complete runs).")

    print("\nGenerating figures...")

    fig_total_cost_line(seeds, a0v, a1sv, a2sv, out_dir)
    fig_savings_vs_a0(seeds, a0v, a1sv, a2sv, out_dir)
    fig_boxplots(a0v, a1sv, a2sv, out_dir)
    fig_win_counts(seeds, a0v, a1sv, a2sv, out_dir)
    fig_chain_analysis(seeds, a0v, a1sv, a2sv, out_dir)
    fig_cdf(a0v, a1sv, a2sv, out_dir)
    fig_pairwise_scatter(a0v, a1sv, a2sv, out_dir)
    fig_rolling_mean(seeds, a0v, a1sv, a2sv, out_dir)
    fig_histogram(a0v, a1sv, a2sv, out_dir)
    fig_summary_card(seeds, a0v, a1sv, a2sv, out_dir)
    fig_per_metric_means(all_metrics, out_dir)
    fig_kpi_heatmap(all_metrics, out_dir)

    print()
    print("=" * 60)
    print("Done. Files written:")
    print(f"  {kpi_path}")
    print(f"  {cost_path}")
    print(f"  + 12 PNG figures in {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
