"""Extract data from overview CSVs.

Overview CSVs (see Testrun_csvs/overview_*.csv) are semicolon-separated
with comma decimals, e.g.:

    target regulator pressure;average pressure before valve;average force reading;read stepper y offset;...
    1,000;0,245;...;-137,300;...

Provides:
    extract_overview_data(csv_path) -> numpy.ndarray
    average_by_target_pressure(data) -> numpy.ndarray
    plot_force_vs_y_offset(datasets, ...) -> (fig, ax)
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Union

import numpy as np

REQUIRED_COLUMNS = [
    "target regulator pressure",
    "average pressure before valve",
    "average force reading",
    "read stepper y offset",
]


def _parse_number(value: str | None) -> float:
    """Parse a number that may use ',' as decimal separator.

    An empty cell (empty string or None) is interpreted as 0.
    """
    if value is None:
        return 0.0
    value = value.strip()
    if value == "":
        return 0.0
    if value.lower() in {"na", "nan", "none", "null", "-"}:
        return float("nan")
    # Overview CSVs use ',' for decimals and have no thousands separator,
    # so a plain comma -> dot replacement is sufficient.
    return float(value.replace(",", "."))


def extract_overview_data(csv_path: Union[str, Path]) -> np.ndarray:
    """Extract the four columns of interest from an overview CSV.

    Args:
        csv_path: Path to an overview CSV file.

    Returns:
        numpy.ndarray of shape (n_rows, 4) with columns in this order:
            0: target regulator pressure
            1: average pressure before valve
            2: average force reading
            3: read stepper y offset

        Rows with missing/unparseable values are skipped.
        Empty cells are interpreted as 0.

    Raises:
        FileNotFoundError: If csv_path does not exist.
        KeyError: If any required column is missing.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        # Sniff delimiter; overview CSVs use ';' but fall back gracefully.
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=";,,\t")
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = ";"

        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"CSV appears to be empty: {csv_path}")

        # Case-insensitive lookup of the required columns.
        lookup = {name.lower().strip(): name for name in reader.fieldnames}
        actual_cols = []
        for required in REQUIRED_COLUMNS:
            key = required.lower().strip()
            if key not in lookup:
                raise KeyError(
                    f"Required column '{required}' not found in {csv_path}. "
                    f"Available columns: {reader.fieldnames}"
                )
            actual_cols.append(lookup[key])

        rows = []
        for line in reader:
            try:
                rows.append([_parse_number(line[col]) for col in actual_cols])
            except (ValueError, TypeError, AttributeError):
                continue  # skip rows with unparseable values

    data = np.array(rows, dtype=float)
    # Drop rows containing NaN (missing values).
    if data.size:
        data = data[~np.isnan(data).any(axis=1)]

    return data.reshape(-1, 4)


def average_by_target_pressure(data: np.ndarray) -> np.ndarray:
    """Average rows sharing the same target pressure, with 95% confidence intervals.

    Args:
        data: numpy.ndarray of shape (n_rows, 4) as returned by
            :func:`extract_overview_data`, with columns:
                0: target regulator pressure
                1: average pressure before valve
                2: average force reading
                3: read stepper y offset

    Returns:
        numpy.ndarray of shape (n_targets, 7) with one row per distinct
        target pressure, sorted ascending by target pressure. Columns:
            0: target regulator pressure
            1: mean pressure before valve
            2: mean force reading
            3: mean stepper y offset
            4: 95% CI half-width of the pressure-before-valve mean
            5: 95% CI half-width of the force-reading mean
            6: sample count n in the group

        The 95% CI half-width is ``t(0.975, n-1) * std / sqrt(n)`` using the
        sample standard deviation (ddof=1). Groups with n <= 1 get a
        half-width of 0.0. Uses scipy's t-distribution when available and
        falls back to the normal approximation (1.96) otherwise.
    """
    data = np.asarray(data, dtype=float)
    if data.size == 0:
        return np.zeros((0, 7), dtype=float)
    if data.ndim != 2 or data.shape[1] != 4:
        raise ValueError(f"Expected array of shape (n, 4), got {data.shape}")

    try:
        from scipy.stats import t as _t_dist

        def _t_value(n: int) -> float:
            if n <= 1:
                return 0.0
            return float(_t_dist.ppf(0.975, n - 1))

    except ImportError:

        def _t_value(n: int) -> float:
            return 1.96 if n > 1 else 0.0

    targets = np.unique(data[:, 0])
    targets.sort()

    out_rows = []
    for target in targets:
        group = data[data[:, 0] == target]
        n = group.shape[0]
        mean_pressure = float(np.mean(group[:, 1]))
        mean_force = float(np.mean(group[:, 2]))
        mean_stepper = float(np.mean(group[:, 3]))
        if n <= 1:
            pressure_ci = 0.0
            force_ci = 0.0
        else:
            t_val = _t_value(n)
            pressure_ci = float(t_val * np.std(group[:, 1], ddof=1) / np.sqrt(n))
            force_ci = float(t_val * np.std(group[:, 2], ddof=1) / np.sqrt(n))
        out_rows.append([float(target), mean_pressure, mean_force, mean_stepper, pressure_ci, force_ci, float(n)])

    return np.array(out_rows, dtype=float)


def plot_force_vs_y_offset(datasets, ax=None, save_path=None, show=False,
                           show_pressures_in_legend=True):
    """Plot mean force with 95% CIs against stepper y offset.

    One colored line per target regulator pressure.

    Args:
        datasets: iterable of numpy arrays, one per y offset (e.g. one per
            overview CSV). Each entry may be either:
                - raw output of :func:`extract_overview_data`, shape (n, 4):
                  [target, pressure, force, stepper_y_offset], or
                - averaged output of :func:`average_by_target_pressure`,
                  shape (n_targets, 7):
                  [target, mean_pressure, mean_force, mean_stepper,
                   pressure_ci, force_ci, n].
        ax: optional matplotlib Axes to plot on. A new figure is created
            when omitted.
        save_path: optional file path to save the figure to.
        show: whether to call ``plt.show()``. Defaults to False so the
            function is headless-safe.
        show_pressures_in_legend: whether to append the per-point mean
            pressures to each legend entry. Disable for many datasets to
            keep the legend compact. Defaults to True (legacy behaviour).

    Returns:
        (fig, ax) matplotlib figure and axes.
    """
    import matplotlib.pyplot as plt

    averaged = []
    for data in datasets:
        arr = np.asarray(data, dtype=float)
        if arr.size == 0:
            continue
        if arr.ndim != 2 or arr.shape[1] not in (4, 7):
            raise ValueError(f"Expected array with 4 or 7 columns, got {arr.shape}")
        if arr.shape[1] == 4:
            arr = average_by_target_pressure(arr)
        averaged.append(arr)

    if not averaged:
        raise ValueError("No data to plot.")

    combined = np.vstack(averaged)  # (total_rows, 7)
    targets = np.unique(combined[:, 0])
    targets.sort()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    for target in targets:
        rows = combined[combined[:, 0] == target]
        order = np.argsort(rows[:, 3])  # sort by mean stepper y offset
        rows = rows[order]
        x = rows[:, 3]
        y = rows[:, 2]
        yerr = rows[:, 5]
        if show_pressures_in_legend:
            pressures = ", ".join(f"{p:.3f}" for p in rows[:, 1])
            label = f"target {target:g} [{pressures}]"
        else:
            label = f"target {target:g}"
        ax.errorbar(
            x, y, yerr=yerr, fmt="o-", capsize=4, markersize=5,
            label=label,
        )

    ax.set_xlabel("Stepper y offset")
    ax.set_ylabel("Mean force reading")
    ax.set_title("Mean force with 95% CI vs y offset (per target pressure)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title="Target pressure")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "Testrun_csvs/overview_20260731_175329_071.csv"
    arr = extract_overview_data(path)
    print(f"shape: {arr.shape}")
    print(arr)
    avg_data = average_by_target_pressure(arr)
    print(f"Average data shape: {avg_data.shape}")
    print(avg_data)
