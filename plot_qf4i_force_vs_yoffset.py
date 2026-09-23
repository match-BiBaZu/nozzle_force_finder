"""Plot mean force vs stepper y offset for the Qf4i testdata overviews.

Loads every ``overview_*.csv`` under one ``Testrun_csvs/testdata_*`` folder
(e.g. ``Testrun_csvs/testdata_Qf4i_pose1+5z/*/``) and uses
:func:`extract_overview_data.plot_force_vs_y_offset` (one line per target
regulator pressure, 95% CIs) to produce the graph.

Each testdata folder covers one pose / z combination, so one graph is drawn
per folder: the pose number and +z info parsed from the folder name
(e.g. ``testdata_Qf4i_pose1+5z`` -> "Qf4i pose 1, +5z") is added to the plot
title and the default output file name, keeping the combinations separate.

Usage:
    .venv/bin/python plot_qf4i_force_vs_yoffset.py --base Testrun_csvs/testdata_Qf4i_pose1+5z
    .venv/bin/python plot_qf4i_force_vs_yoffset.py --base Testrun_csvs/testdata_Qf4i_pose1+5z --out my_plot.png
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

from extract_overview_data import extract_overview_data, plot_force_vs_y_offset

DEFAULT_BASE = "Testrun_csvs/testdata__Qf4i_pose7+1z"

# e.g. testdata_Qf4i_pose1+5z or testdata__Qf4i_pose7+1z -> ("1", "5") / ("7", "1")
POSE_Z_PATTERN = re.compile(r"pose(\d+)\+(\d+)z", re.IGNORECASE)


def pose_z_parts(folder_name: str) -> tuple[str, str]:
    """Derive (label, slug) from a testdata folder name.

    E.g. ``testdata_Qf4i_pose1+5z`` -> ("Qf4i pose 1, +5z", "pose1+5z").
    Falls back to the raw folder name when the pattern doesn't match.
    """
    match = POSE_Z_PATTERN.search(folder_name)
    if match:
        pose, z = int(match.group(1)), int(match.group(2))
        return f"Qf4i pose {pose}, +{z}z", f"pose{pose}+{z}z"
    return folder_name, folder_name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=DEFAULT_BASE,
                        help="Testdata folder with one impulse subfolder per "
                             "y offset (each containing its overview_*.csv).")
    parser.add_argument("--out", default=None,
                        help="Output image path. Defaults to "
                             "force_vs_yoffset_qf4i_<pose>+<z>z.png.")
    args = parser.parse_args(argv)

    base = Path(args.base)
    label, slug = pose_z_parts(base.name)
    out = args.out or f"force_vs_yoffset_qf4i_{slug}.png"

    paths = sorted(str(p) for p in base.glob("*/overview_*.csv"))
    if not paths:
        print(f"Error: no overview CSVs found in {base}/*/overview_*.csv",
              file=sys.stderr)
        return 1

    datasets = []
    for path in paths:
        data = extract_overview_data(path)
        if data.size == 0:
            print(f"WARNING: no usable rows in {path}, skipping.")
            continue
        print(f"{Path(path).parent.name}: {data.shape[0]} impulses")
        datasets.append(data)

    if not datasets:
        print("Error: no data to plot.", file=sys.stderr)
        return 1

    fig, ax = plot_force_vs_y_offset(datasets, save_path=None,
                                     show_pressures_in_legend=False)
    ax.set_title(f"Mean force with 95% CI vs y offset ({label}, per target pressure)")
    # Several target pressures: move the legend outside so it doesn't cover data.
    legend = ax.get_legend()
    if legend is not None:
        legend.set_loc("upper left")
        legend.set_bbox_to_anchor((1.02, 1.0))
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"Saved plot to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
