"""force_pressure_graph.py

Read a testrun CSV and plot averaged force vs averaged pressure-before-valve
for each distinct target regulator pressure used in the testrun.

Provides function:
    graph_pressure_force_from_csv(csv_path, ax=None, save_path=None, show=True)

- Finds relevant columns (case-insensitive, permissive matching):
    * target regulator pressure
    * average pressure before valve
    * average force reading (or similar)
- Groups rows by target regulator pressure, computes the mean of the other two
  columns per-group and plots mean force (y) vs mean "average pressure before valve" (x).

Usage as CLI:
    python force_pressure_graph.py path/to/testrun.csv --out out.png

"""

from __future__ import annotations
import argparse
import sys
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


_POSSIBLE_TARGET_COLS = [
    "target regulator pressure",
    "target regulator pressure (psi)",
    "target regulator pressure (bar)",
    "target_pressure",
    "target regulator",
]

_POSSIBLE_PRESSURE_BEFORE_VALVE_COLS = [
    "average pressure before valve",
    "avg pressure before valve",
    "average_pressure_before_valve",
    "pressure before valve",
    "pressure_before_valve",
    "avg_pressure_before_valve",
]

_POSSIBLE_FORCE_COLS = [
    "average force reading",
    "avg force reading",
    "average_force_reading",
    "avg_force",
    "average_force",
    "force",
]


def _find_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    """Find the first candidate that matches a column (case-insensitive, stripped).

    Returns the actual column name from columns or None.
    """
    lc_map = {c.lower().strip(): c for c in columns}
    for cand in candidates:
        cand_lc = cand.lower().strip()
        if cand_lc in lc_map:
            return lc_map[cand_lc]
    # try fuzzy: remove spaces/underscores and compare
    def normalize(s: str) -> str:
        return s.lower().replace(" ", "").replace("_", "")

    norm_map = {normalize(c): c for c in columns}
    for cand in candidates:
        nc = normalize(cand)
        if nc in norm_map:
            return norm_map[nc]
    # last resort: try partial contains
    for cand in candidates:
        cand_lc = cand.lower().strip()
        for col in columns:
            if cand_lc in col.lower():
                return col
    return None


def graph_pressure_force_from_csv(csv_path: str,
                                  ax: Optional[plt.Axes] = None,
                                  save_path: Optional[str] = None,
                                  show: bool = True) -> Tuple[pd.DataFrame, plt.Figure]:
    """Read the CSV at csv_path and plot averaged force vs pressure-before-valve.

    Returns (grouped_df, fig).

    grouped_df columns will include:
        - target_regulator_pressure (the grouped-by value)
        - mean_avg_pressure_before_valve (x coordinate)
        - mean_force (y coordinate)

    The function will:
      - read CSV with pandas
      - locate the three relevant columns using permissive matching
      - drop rows where any of the three columns are missing
      - group by target regulator pressure and compute means
      - sort by mean pressure-before-valve and plot mean_force vs mean_pressure_before_valve

    """
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"CSV appears to be empty: {csv_path}")

    cols = list(df.columns)

    target_col = _find_column(cols, _POSSIBLE_TARGET_COLS)
    pressure_before_col = _find_column(cols, _POSSIBLE_PRESSURE_BEFORE_VALVE_COLS)
    force_col = _find_column(cols, _POSSIBLE_FORCE_COLS)

    missing = []
    if target_col is None:
        missing.append("target regulator pressure")
    if pressure_before_col is None:
        missing.append("average pressure before valve")
    if force_col is None:
        missing.append("average force reading")
    if missing:
        raise KeyError(f"Could not find required columns in CSV. Missing: {', '.join(missing)}.\nAvailable columns: {cols}")

    # Coerce to numeric where possible
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df[pressure_before_col] = pd.to_numeric(df[pressure_before_col], errors="coerce")
    df[force_col] = pd.to_numeric(df[force_col], errors="coerce")

    # Drop rows missing any of the three
    df_clean = df.dropna(subset=[target_col, pressure_before_col, force_col])

    if df_clean.empty:
        raise ValueError("No rows remain after dropping NaNs from required columns.")

    # Group by target regulator pressure and compute mean of the other two
    grouped = (
        df_clean
        .groupby(target_col, as_index=False)
        .agg(
            mean_pressure_before_valve=(pressure_before_col, "mean"),
            mean_force=(force_col, "mean"),
            count=(force_col, "count"),
            std_force=(force_col, "std"),
        )
    )

    # Sort by mean_pressure_before_valve to make a sensible x ordering on the plot
    grouped = grouped.sort_values(by="mean_pressure_before_valve")

    # Plot
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    x = grouped["mean_pressure_before_valve"].to_numpy()
    y = grouped["mean_force"].to_numpy()
    errs = grouped["std_force"].to_numpy()

    ax.errorbar(x, y, yerr=errs, fmt="o-", linewidth=1.5, markersize=6, capsize=4)
    ax.set_xlabel("Average pressure before valve")
    ax.set_ylabel("Average force reading")
    ax.set_title("Average force vs average pressure-before-valve (per target regulator pressure)")
    ax.grid(True, linestyle='--', alpha=0.4)

    # annotate each point with the target regulator pressure value (group key)
    for xi, yi, key in zip(x, y, grouped[target_col]):
        ax.annotate(str(key), (xi, yi), textcoords="offset points", xytext=(4, 4), fontsize=8)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200)

    if show:
        plt.show()
    else:
        plt.close(fig)

    # Prepare a clean dataframe to return with consistent column names
    out_df = grouped.rename(columns={target_col: "target_regulator_pressure"})

    return out_df, fig


def _cli(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Plot averaged force vs pressure-before-valve from a testrun CSV.")
    p.add_argument("csv", help="Path to the testrun CSV file")
    p.add_argument("--out", "-o", help="Output image path (optional). If omitted the plot will be shown but not saved.")
    p.add_argument("--no-show", action="store_true", help="Do not call plt.show(); useful when running headless")
    args = p.parse_args(argv)

    try:
        df, fig = graph_pressure_force_from_csv(args.csv, save_path=args.out, show=not args.no_show)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    if args.out:
        print(f"Saved plot to {args.out}")
    else:
        print("Plotting completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
