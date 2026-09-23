"""Build overview CSVs for the Qf4i testdata impulse folders.

Each subfolder of ``Testrun_csvs/testdata__Qf4i_pose7+1z/`` (e.g.
``impulse_Qf4i_pose7_0x+1z/``) contains one ``*_summary.csv`` / ``*_timeseries.csv``
pair per impulse. The per-impulse summary files are key-value tables::

    parameter;value;unit / note
    sequence id;20260918_135731_328;
    ...

This script aggregates the ``*_summary.csv`` files of each subfolder into a
single wide overview CSV with the same layout as the legacy overviews in
``Testrun_csvs/old/overview_*.csv``::

    sequence id;impulse id;recorded at local;timeseries file;summary file;...
    20260918_135731_328;20260918_135731_328_i0001;...;...

The output uses ``;`` delimiters, comma decimals and utf-8-sig encoding, so it
is directly compatible with :func:`extract_overview_data.extract_overview_data`
(which requires the columns ``target regulator pressure``,
``average pressure before valve``, ``average force reading`` and
``read stepper y offset``).

Column mapping notes (new summary key -> overview column):
    - ``plc valve open duration ms`` -> ``arduino valve open duration ms``
      (renamed; old files predate the PLC naming).
    - ``nozzle 5 used`` / ``nozzle 6 used`` and the snake_case raw-value
      duplicates (``baseline_force_n`` etc.) have no counterpart in the legacy
      header and are dropped.
    - ``timeseries file`` / ``summary file`` are filled with the sibling file
      names found in the impulse folder.
    - All other values are copied verbatim (preserving comma decimals and
      empty cells for missing values). Rows are sorted by impulse index.

Usage:
    python build_qf4i_overviews.py [--base DIR] [--output-dir DIR]

Defaults:
    --base Testrun_csvs/testdata__Qf4i_pose7+1z
    output: overview_<subfolder>.csv written into each subfolder
    (use --output-dir to collect them all in one directory instead).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Exact header of the legacy overviews in Testrun_csvs/old/overview_*.csv.
OVERVIEW_HEADER = [
    "sequence id",
    "impulse id",
    "recorded at local",
    "timeseries file",
    "summary file",
    "calibration profile id",
    "impulse index",
    "flip angle",
    "pose number",
    "hole number",
    "read stepper y offset",
    "read colibri z offset",
    "start time ms",
    "valve close time ms",
    "capture end time ms",
    "arduino valve open duration ms",
    "target regulator pressure",
    "average actual regulator pressure",
    "average pressure before valve",
    "average force reading",
    "average force 1",
    "average force 2",
    "maximum flow",
    "volume l",
    "nozzle 1 used",
    "nozzle 2 used",
    "nozzle 3 used",
    "nozzle 4 used",
    "valve open sample count",
    "pressure sample count",
    "flow sample count",
    "force baseline n",
    "force baseline standard deviation n",
    "peak force n",
    "time to peak ms",
    "force rise 10-90 ms",
    "force fall 90-10 ms",
    "force fwhm ms",
    "force impulse ns",
    "force plateau mean n",
    "force plateau standard deviation n",
    "actual minus target pressure bar",
    "peak force per actual pressure n per bar",
    "force 1 share percent",
    "force 2 share percent",
]

# New key name -> legacy overview column name for the renamed columns.
RENAMED_KEYS = {
    "plc valve open duration ms": "arduino valve open duration ms",
}

# Display field -> raw snake_case duplicate used as fallback when the display
# field is empty. The per-impulse summaries leave e.g.
# "average pressure before valve" empty while "pressure_before_valve_bar"
# holds the measurement (with dot decimals); falling back keeps the overview
# pressure columns useful instead of empty.
FALLBACK_KEYS = {
    "target regulator pressure": "target_pressure_bar",
    "average actual regulator pressure": "actual_pressure_bar",
    "average pressure before valve": "pressure_before_valve_bar",
}


def _to_comma_decimals(value: str) -> str:
    """Convert a dot-decimal number string to the comma-decimal format."""
    value = value.strip()
    if value and "." in value and "," not in value:
        return value.replace(".", ",")
    return value


def read_summary_kv(path: Path) -> dict[str, str]:
    """Read a per-impulse ``*_summary.csv`` key-value file.

    Returns a dict mapping parameter name -> raw value string (verbatim, so
    comma decimals and empty cells are preserved).
    """
    mapping: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        if reader.fieldnames is None:
            return mapping
        for row in reader:
            param = (row.get("parameter") or "").strip()
            if not param:
                continue
            value = row.get("value", "")
            if value is None:
                value = ""
            mapping[param] = value
    # Apply renames (e.g. plc -> arduino valve duration) without
    # overwriting a natively-present legacy key.
    for new_key, legacy_key in RENAMED_KEYS.items():
        if new_key in mapping and legacy_key not in mapping:
            mapping[legacy_key] = mapping[new_key]
    return mapping


def build_overview_rows(folder: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Collect one overview row per ``*_summary.csv`` in *folder*.

    Returns (rows, skipped) where rows are dicts keyed by OVERVIEW_HEADER
    column and skipped lists file names that could not be parsed.
    """
    summary_files = sorted(folder.glob("*_summary.csv"))
    rows: list[dict[str, str]] = []
    skipped: list[str] = []
    for summary_path in summary_files:
        try:
            kv = read_summary_kv(summary_path)
        except (OSError, csv.Error):
            skipped.append(summary_path.name)
            continue
        if not kv.get("impulse id"):
            skipped.append(summary_path.name)
            continue
        # Sibling timeseries file: same basename with _timeseries.csv suffix.
        timeseries_name = summary_path.name.replace("_summary.csv", "_timeseries.csv")
        if not (folder / timeseries_name).is_file():
            timeseries_name = ""
        row = {col: kv.get(col, "") for col in OVERVIEW_HEADER}
        for col, fallback_key in FALLBACK_KEYS.items():
            if not row[col].strip() and kv.get(fallback_key, "").strip():
                row[col] = _to_comma_decimals(kv[fallback_key])
        row["timeseries file"] = timeseries_name
        row["summary file"] = summary_path.name
        rows.append(row)

    def _index_key(row: dict[str, str]) -> tuple[int, str]:
        try:
            return (int(row.get("impulse index", "") or 0), row.get("impulse id", ""))
        except ValueError:
            return (0, row.get("impulse id", ""))

    rows.sort(key=_index_key)
    return rows, skipped


def write_overview_csv(rows: list[dict[str, str]], dest: Path) -> None:
    """Write *rows* with the legacy overview header/formatting."""
    with dest.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, delimiter=";", fieldnames=OVERVIEW_HEADER,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_all_overviews(base: Path, output_dir: Path | None = None) -> list[Path]:
    """Build one overview CSV per immediate subfolder of *base*.

    Returns the list of written overview paths.
    """
    if not base.is_dir():
        raise FileNotFoundError(f"Base directory not found: {base}")
    subfolders = sorted(p for p in base.iterdir() if p.is_dir())
    if not subfolders:
        raise FileNotFoundError(f"No subfolders found in: {base}")
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for folder in subfolders:
        rows, skipped = build_overview_rows(folder)
        if not rows:
            print(f"WARNING: no impulses found in {folder}, skipping.", file=sys.stderr)
            continue
        dest = (output_dir / f"overview_{folder.name}.csv") if output_dir \
            else (folder / f"overview_{folder.name}.csv")
        write_overview_csv(rows, dest)
        written.append(dest)
        msg = f"Wrote {dest} ({len(rows)} rows)"
        if skipped:
            msg += f" [skipped {len(skipped)}: {', '.join(skipped)}]"
        print(msg)
    return written


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="Testrun_csvs/testdata__Qf4i_pose7+1z",
                        help="Directory containing per-pose impulse subfolders.")
    parser.add_argument("--output-dir", default=None,
                        help="Collect all overview CSVs in this directory "
                             "instead of writing one per subfolder.")
    args = parser.parse_args(argv)
    try:
        build_all_overviews(Path(args.base),
                            Path(args.output_dir) if args.output_dir else None)
    except (FileNotFoundError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
