from __future__ import annotations

from pathlib import Path

from _rq1_iter5_model_vs_baselines_from_log_template import generate_one_model_figure


def main() -> None:
    root = Path(__file__).resolve().parent
    generate_one_model_figure(
        ml="GBR",
        log_path=root / "rq1_iter5_metrics.log",
        out_dir=root / "figures_rq1_iter5_from_log",
        ml_color="#e7298a",
    )
    print("Done: GBR")


if __name__ == "__main__":
    main()

