"""
Extract Iteration-5 regression metrics from the LaTeX table (table_complete_regression.tex)
and write a simple CSV-like log file used by plotting scripts.

Input:
  - src/table_complete_regression.tex

Output:
  - src/rq1_iter5_metrics.log   (CSV with header)

The parser is intentionally strict for this repository's table format:
  - project starts on lines with \\multirow{7}{*}{<project>}
  - subsequent 6 lines start with '& <Model> & ...'
"""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TABLE_TEX = ROOT / "table_complete_regression.tex"
OUT_LOG = ROOT / "rq1_iter5_metrics.log"


def _clean_cell(s: str) -> str:
    s = s.strip()
    s = s.replace(r"\textbf{", "").replace("}", "")
    s = s.replace(r"\,", "")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def main() -> None:
    if not TABLE_TEX.exists():
        raise SystemExit(f"Missing input file: {TABLE_TEX}")

    txt = TABLE_TEX.read_text(encoding="utf-8", errors="replace").splitlines()

    rows: list[tuple[str, str, float, float, float, float]] = []
    current_project: str | None = None

    # Example lines:
    # \multirow{7}{*}{radare2} & RF & \textbf{1096.37} & 0.80 & \textbf{0.35} & 0.47 \\
    #  & LGB & 1137.32 & 0.80 & 0.34 & 0.47 \\
    proj_re = re.compile(r"\\multirow\{7\}\{\*\}\{([^}]+)\}\s*&\s*(.+?)\s*\\\\")
    cont_re = re.compile(r"^\s*&\s*(.+?)\s*\\\\")

    for line in txt:
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        if line.startswith(r"\midrule") or line.startswith(r"\toprule") or line.startswith(r"\bottomrule"):
            continue

        m = proj_re.search(line)
        if m:
            current_project = _clean_cell(m.group(1))
            payload = m.group(2)
        else:
            m2 = cont_re.search(line)
            if not m2:
                continue
            if current_project is None:
                raise SystemExit("Found continuation row before first project.")
            payload = m2.group(1)

        # payload is: Model & RMSE & NRMSE & R2 & CVRMSE
        parts = [p.strip() for p in payload.split("&")]
        if len(parts) != 5:
            continue

        model = _clean_cell(parts[0])
        rmse = float(_clean_cell(parts[1]))
        nrmse = float(_clean_cell(parts[2]))
        r2 = float(_clean_cell(parts[3]))
        cvrmse = float(_clean_cell(parts[4]).replace(r"\\", "").strip())
        rows.append((current_project, model, rmse, nrmse, r2, cvrmse))

    # basic sanity check: 10 projects * 7 models = 70 rows
    if len(rows) < 70:
        raise SystemExit(f"Parsed only {len(rows)} rows; expected 70. Check table format.")

    out_lines = ["project,model,RMSE,NRMSE,R2,CVRMSE"]
    for project, model, rmse, nrmse, r2, cvrmse in rows:
        out_lines.append(f"{project},{model},{rmse:.2f},{nrmse:.2f},{r2:.2f},{cvrmse:.2f}")

    OUT_LOG.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print("Wrote:", OUT_LOG)


if __name__ == "__main__":
    main()

