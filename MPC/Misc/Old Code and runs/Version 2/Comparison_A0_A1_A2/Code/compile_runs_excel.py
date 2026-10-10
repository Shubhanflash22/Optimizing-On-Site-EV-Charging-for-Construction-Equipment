"""
compile_runs_excel.py
=====================
Reads   Output_AllRuns/<label>/live_data/A0_A1S_A2S/A0_vs_A1S_vs_A2S.html
for every one of the 25 runs defined in RUN_ALL_25_RUNS.jl and compiles
the results into a single Excel workbook.

Layout (matches Sample.xlsx):
  Row 1  : run labels, merged over the 3 approach columns (OS | CE | SB)
  Row 2  : "Metric" | OS | CE | SB | OS | CE | SB | …
  Row 3+ : one metric per row

Runs whose HTML file is missing are flagged "run missing" in row 1 and
left blank in the data rows — no error is raised.

Usage:
    python compile_runs_excel.py [OUTPUT_ALLRUNS_DIR] [OUTPUT_XLSX]

Defaults:
    OUTPUT_ALLRUNS_DIR = C:\\Users\\shubh\\Downloads\\To be copied\\Output_AllRuns
    OUTPUT_XLSX        = C:\\Users\\shubh\\Downloads\\To be copied\\MPC_Comparison_AllRuns.xlsx
"""

import os, sys, re
from pathlib import Path
from html.parser import HTMLParser

import openpyxl
from openpyxl.styles import (Font, PatternFill, Alignment, Border, Side,
                              GradientFill)
from openpyxl.utils import get_column_letter

# ---------------------------------------------------------------------------
# 1. Run definitions (mirrors RUN_ALL_25_RUNS.jl) — only label & mode matter
#    for building the file path.
# ---------------------------------------------------------------------------
RUNS = [
    # ---- 3-seed block @ 1200s, 1 day (runs 01-15) ----
    dict(label="01_GroundTruth_seed7_1200s",  mode="live_data"),
    dict(label="02_GroundTruth_seed18_1200s", mode="live_data"),
    dict(label="03_GroundTruth_seed55_1200s", mode="live_data"),
    dict(label="04_High_seed7_1200s",         mode="high"),
    dict(label="05_High_seed18_1200s",        mode="high"),
    dict(label="06_High_seed55_1200s",        mode="high"),
    dict(label="07_Low_seed7_1200s",          mode="low"),
    dict(label="08_Low_seed18_1200s",         mode="low"),
    dict(label="09_Low_seed55_1200s",         mode="low"),
    dict(label="10_NearMean_seed7_1200s",     mode="near_mean"),
    dict(label="11_NearMean_seed18_1200s",    mode="near_mean"),
    dict(label="12_NearMean_seed55_1200s",    mode="near_mean"),
    dict(label="13_Normal_seed7_1200s",       mode="normal"),
    dict(label="14_Normal_seed18_1200s",      mode="normal"),
    dict(label="15_Normal_seed55_1200s",      mode="normal"),
    # ---- 5-day block, seed 7, 1200s (runs 16-20) ----
    dict(label="16_GroundTruth_seed7_5days",  mode="live_data"),
    dict(label="17_High_seed7_5days",         mode="high"),
    dict(label="18_Low_seed7_5days",          mode="low"),
    dict(label="19_NearMean_seed7_5days",     mode="near_mean"),
    dict(label="20_Normal_seed7_5days",       mode="normal"),
    # ---- 1-hour (3600s) block, seed 7, 1 day (runs 21-25) ----
    dict(label="21_GroundTruth_seed7_3600s",  mode="live_data"),
    dict(label="22_High_seed7_3600s",         mode="high"),
    dict(label="23_Low_seed7_3600s",          mode="low"),
    dict(label="24_NearMean_seed7_3600s",     mode="near_mean"),
    dict(label="25_Normal_seed7_3600s",       mode="normal"),
]

# Column order in the HTML: A0 → OS, A1S → CE, A2S → SB
APPROACH_HEADERS = ["OS", "CE", "SB"]  # one per data column inside a run

# Metric display order (as they appear in the HTML)
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

# ---------------------------------------------------------------------------
# 2. HTML parser — extracts the table rows from A0_vs_A1S_vs_A2S.html
# ---------------------------------------------------------------------------
class TableParser(HTMLParser):
    """Pull every <tr> out of the first <table> found in the file."""

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
    """
    Returns {metric_name: [A0_value, A1S_value, A2S_value]}
    Skips the header row.
    """
    parser = TableParser()
    parser.feed(path.read_text(encoding="utf-8"))

    result = {}
    for row in parser.rows[1:]:          # skip the header row
        if len(row) >= 4:
            metric = row[0]
            values = row[1:4]            # A0, A1S, A2S
            result[metric] = values
    return result


# ---------------------------------------------------------------------------
# 3. Resolve the HTML path for a given run
#    The sub-folder depends on the mode keyword:
#      :live_data  → live_data/A0_A1S_A2S/
#      :high/:low/:near_mean/:normal → <mode>/A0_A1S_A2S/
# ---------------------------------------------------------------------------
MODE_SUBDIR = {
    "live_data":  "live_data",
    "high":       "high",
    "low":        "low",
    "near_mean":  "near_mean",
    "normal":     "normal",
}

def html_path(root: Path, run: dict) -> Path:
    sub = MODE_SUBDIR.get(run["mode"], run["mode"])
    return root / run["label"] / sub / "A0_A1S_A2S" / "A0_vs_A1S_vs_A2S.html"


# ---------------------------------------------------------------------------
# 4. Styling helpers
# ---------------------------------------------------------------------------
FONT_NAME = "Arial"

# Header fills
FILL_RUN_HEADER  = PatternFill("solid", fgColor="1F4E79")   # dark blue — run label row
FILL_COL_HEADER  = PatternFill("solid", fgColor="2E75B6")   # medium blue — OS/CE/SB row
FILL_METRIC_HEADER = PatternFill("solid", fgColor="D6E4F0") # very light blue — metric name cells

# Approach column fills (alternating per run group)
FILL_OS = PatternFill("solid", fgColor="EBF3FB")   # lightest blue
FILL_CE = PatternFill("solid", fgColor="D6E4F0")   # light blue
FILL_SB = PatternFill("solid", fgColor="BDD7EE")   # a touch darker

# Total row fill
FILL_TOTAL = PatternFill("solid", fgColor="FCE4D6")  # light orange

# Missing run fill
FILL_MISSING = PatternFill("solid", fgColor="FFE7E7")  # light red

THIN = Side(style="thin", color="AAAAAA")
MED  = Side(style="medium", color="666666")

def thin_border():
    return Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

def medium_border():
    return Border(left=MED, right=MED, top=MED, bottom=MED)

def style_cell(cell, *, bold=False, italic=False, size=10, color="000000",
               fill=None, halign="center", valign="center",
               border=None, wrap=False, num_format=None):
    cell.font = Font(name=FONT_NAME, bold=bold, italic=italic,
                     size=size, color=color)
    cell.alignment = Alignment(horizontal=halign, vertical=valign,
                               wrap_text=wrap)
    if fill:
        cell.fill = fill
    if border:
        cell.border = border
    if num_format:
        cell.number_format = num_format


# ---------------------------------------------------------------------------
# 5. Build the workbook
# ---------------------------------------------------------------------------
def build_workbook(root: Path, out_path: Path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "All Runs"

    n_runs = len(RUNS)
    n_approaches = len(APPROACH_HEADERS)   # 3

    # ------------------------------------------------------------------
    # Column layout:
    #   col A  = Metric label
    #   cols B onward: groups of 3 (OS, CE, SB) per run
    # ------------------------------------------------------------------
    def run_col_start(run_idx: int) -> int:
        """1-based column index of the first approach column for run_idx."""
        return 2 + run_idx * n_approaches   # col B = 2

    # ------------------------------------------------------------------
    # Row 1 — run labels (merged across 3 approach columns)
    # ------------------------------------------------------------------
    # Cell A1 — corner label
    ws.cell(row=1, column=1, value="Run / Metric")
    style_cell(ws.cell(row=1, column=1),
               bold=True, size=11, color="FFFFFF",
               fill=FILL_RUN_HEADER, border=thin_border())

    for i, run in enumerate(RUNS):
        col_s = run_col_start(i)
        col_e = col_s + n_approaches - 1
        html = html_path(root, run)

        # Merge the 3 approach cells for this run's label
        ws.merge_cells(start_row=1, start_column=col_s,
                       end_row=1,   end_column=col_e)
        cell = ws.cell(row=1, column=col_s)

        if html.exists():
            cell.value = run["label"]
            style_cell(cell, bold=True, size=10, color="FFFFFF",
                       fill=FILL_RUN_HEADER, halign="center", border=thin_border())
        else:
            cell.value = f"run missing\n({run['label']})"
            style_cell(cell, bold=True, italic=True, size=9, color="C00000",
                       fill=FILL_MISSING, halign="center", border=thin_border(), wrap=True)

    # ------------------------------------------------------------------
    # Row 2 — approach sub-headers (Metric | OS CE SB | OS CE SB | …)
    # ------------------------------------------------------------------
    ws.cell(row=2, column=1, value="Metric")
    style_cell(ws.cell(row=2, column=1),
               bold=True, size=10, color="FFFFFF",
               fill=FILL_COL_HEADER, border=thin_border())

    APPROACH_FILLS = [FILL_OS, FILL_CE, FILL_SB]
    for i in range(n_runs):
        col_s = run_col_start(i)
        for j, hdr in enumerate(APPROACH_HEADERS):
            cell = ws.cell(row=2, column=col_s + j, value=hdr)
            style_cell(cell, bold=True, size=10, color="FFFFFF",
                       fill=FILL_COL_HEADER, border=thin_border())

    # ------------------------------------------------------------------
    # Collect data for each run
    # ------------------------------------------------------------------
    run_data: list[dict | None] = []   # None = missing
    for run in RUNS:
        html = html_path(root, run)
        if html.exists():
            try:
                run_data.append(parse_html(html))
            except Exception as e:
                print(f"  [WARN] Could not parse {html}: {e}")
                run_data.append(None)
        else:
            print(f"  [INFO] run missing: {html}")
            run_data.append(None)

    # ------------------------------------------------------------------
    # Rows 3+ — one row per metric
    # ------------------------------------------------------------------
    for m_idx, metric in enumerate(EXPECTED_METRICS):
        row = 3 + m_idx
        is_total = "TOTAL" in metric.upper()

        # Metric label in column A
        label_cell = ws.cell(row=row, column=1, value=metric)
        style_cell(label_cell,
                   bold=is_total, size=10, color="1F3864",
                   fill=(FILL_TOTAL if is_total else FILL_METRIC_HEADER),
                   halign="left", border=thin_border())

        # Data cells
        for i, (run, data) in enumerate(zip(RUNS, run_data)):
            col_s = run_col_start(i)
            for j in range(n_approaches):
                cell = ws.cell(row=row, column=col_s + j)
                if data is None:
                    # run missing — leave blank but mark with a very light red
                    cell.value = ""
                    fill = FILL_MISSING
                else:
                    values = data.get(metric, ["", "", ""])
                    raw = values[j] if j < len(values) else ""
                    # Try to store as a number for proper alignment / formatting
                    try:
                        cell.value = float(raw)
                    except (ValueError, TypeError):
                        cell.value = raw
                    fill = FILL_TOTAL if is_total else APPROACH_FILLS[j]

                style_cell(cell,
                           bold=is_total, size=10,
                           fill=fill, border=thin_border(),
                           num_format='#,##0.000' if not is_total else '#,##0.000')

    # ------------------------------------------------------------------
    # Column widths
    # ------------------------------------------------------------------
    ws.column_dimensions["A"].width = 34   # metric labels
    for i in range(n_runs):
        col_s = run_col_start(i)
        for j in range(n_approaches):
            col_letter = get_column_letter(col_s + j)
            ws.column_dimensions[col_letter].width = 12

    # Row heights
    ws.row_dimensions[1].height = 32   # run label row (may wrap)
    ws.row_dimensions[2].height = 18   # approach header

    # Freeze panes: keep row 1+2 and col A visible while scrolling
    ws.freeze_panes = "B3"

    # ------------------------------------------------------------------
    # Legend sheet
    # ------------------------------------------------------------------
    ws_leg = wb.create_sheet("Legend")
    legend_rows = [
        ("Abbreviation", "Full name", "Description"),
        ("OS", "One-Shot Naïve Optimization", "Approach 0 — single-shot optimisation (A0)"),
        ("CE", "Certainty Equivalent MPC",     "Approach 1 — shrinking-horizon MPC using expected values (A1S)"),
        ("SB", "Scenario-Based MPC",           "Approach 2 — shrinking-horizon stochastic / scenario-based MPC (A2S)"),
        ("", "", ""),
        ("run missing", "", "HTML result file not found for that run; data cells are left blank"),
    ]
    for r_idx, leg_row in enumerate(legend_rows, start=1):
        for c_idx, val in enumerate(leg_row, start=1):
            cell = ws_leg.cell(row=r_idx, column=c_idx, value=val)
            is_hdr = r_idx == 1
            style_cell(cell,
                       bold=is_hdr, size=10,
                       fill=(FILL_RUN_HEADER if is_hdr else None),
                       color=("FFFFFF" if is_hdr else "000000"),
                       halign="left", border=thin_border())
    ws_leg.column_dimensions["A"].width = 14
    ws_leg.column_dimensions["B"].width = 36
    ws_leg.column_dimensions["C"].width = 60

    wb.save(out_path)
    print(f"\n✅  Saved: {out_path}")


# ---------------------------------------------------------------------------
# 6. Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    default_root = Path(r"C:\Users\shubh\Downloads\To be copied\Output_AllRuns")
    default_out  = Path(r"C:\Users\shubh\Downloads\To be copied\MPC_Comparison_AllRuns.xlsx")

    root    = Path(sys.argv[1]) if len(sys.argv) > 1 else default_root
    out_xls = Path(sys.argv[2]) if len(sys.argv) > 2 else default_out

    print(f"Output_AllRuns root : {root}")
    print(f"Output Excel        : {out_xls}")
    print(f"Runs to process     : {len(RUNS)}")
    print("-" * 60)

    if not root.exists():
        print(f"[WARN] Root directory not found: {root}")
        print("       Running in demo mode — all runs will show 'run missing'.")

    build_workbook(root, out_xls)
