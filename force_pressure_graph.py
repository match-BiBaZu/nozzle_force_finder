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


# Linear fit: average pressure before valve [bar] from target regulator
# pressure [bar], fitted on target_vs_nozzle_pressure_data (qf4i_p6_h0..h3,
# 520 points, targets 1.5..4.0 bar):
#     P_before = SLOPE * P_target + INTERCEPT
# Combined OLS: slope=0.502929, intercept=0.012125, R2=0.9932, RMSE=0.031 bar.
TARGET_TO_VALVE_SLOPE = 0.5029292307692308
TARGET_TO_VALVE_INTERCEPT = 0.012125384615384396


def target_to_valve_pressure_bar(target_bar: float,
                                 slope: float = TARGET_TO_VALVE_SLOPE,
                                 intercept: float = TARGET_TO_VALVE_INTERCEPT) -> float:
    """Map target regulator pressure [bar] to nozzle (before-valve) pressure [bar].

    Uses the linear fit P_before = slope * P_target + intercept.
    """
    return float(slope * float(target_bar) + intercept)


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


def graph_simulated_force_vs_yoffset(y_values=None,
                                     x: float = -25.2,
                                     z: float = 0.0,
                                     name_obj: str = "Qf4i",
                                     cw: float = 1,
                                     nozzle_distance: float = 17.5,
                                     nozzle_diameter: float = 0.8,
                                     nozzle_pressure: float = 300000,
                                     mesh_coefficient: float = 0.49,
                                     ax: Optional[plt.Axes] = None,
                                     save_path: Optional[str] = None,
                                     show: bool = True,
                                     **fan_kwargs) -> Tuple[pd.DataFrame, plt.Figure]:
    """Plot simulated force vs y offset using NozzleForceFan2.calc_force_fan.

    For each y in y_values calls::

        calc_force_fan(name_obj, [x, y, z], cw, nozzle_distance,
                       nozzle_diameter, nozzle_pressure)

    Default sweep is y = -10, -20, ..., -70 (steps of 10), x = -25.2, z = 0.

    Extra kwargs are forwarded to calc_force_fan (e.g. ray_number,
    num_nozzles, nozzle_offset, cone_divisions). print_results/graph/use_gui
    default to False unless overridden. mesh_coefficient (default 0.49) is
    applied inside calc_force_fan to scale the simulated forces.

    Returns (df, fig) where df has columns [y_offset, total_force_n].
    """
    from NozzleForceFan2 import calc_force_fan

    if y_values is None:
        y_values = list(range(-10, -71, -10))
    y_values = list(y_values)

    fan_kwargs.setdefault("print_results", False)
    fan_kwargs.setdefault("graph", False)
    fan_kwargs.setdefault("use_gui", False)
    fan_kwargs.setdefault("mesh_coefficient", mesh_coefficient)

    forces = []
    for y in y_values:
        total_force, *_ = calc_force_fan(
            name_obj, [x, y, z], cw, nozzle_distance, nozzle_diameter,
            nozzle_pressure, **fan_kwargs)
        forces.append(float(total_force))

    df = pd.DataFrame({"y_offset": y_values, "total_force_n": forces})
    df = df.sort_values(by="y_offset")

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    ax.plot(df["y_offset"].to_numpy(), df["total_force_n"].to_numpy(), "o-",
            linewidth=1.5, markersize=6)
    ax.set_xlabel("Y offset [mm]")
    ax.set_ylabel("Simulated total force [N]")
    ax.set_title(f"Simulated force vs y offset ({name_obj}, "
                 f"{nozzle_pressure / 1e5:.1f} bar, mesh_coeff={mesh_coefficient})")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return df, fig


def graph_testdata_vs_simulated(testdata_sources,
                                y_values=None,
                                x: float = -25.2,
                                z: float = 0.0,
                                name_obj: str = "Qf4i",
                                cw: float = 1,
                                nozzle_distance: float = 17.5,
                                nozzle_diameter: float = 0.8,
                                nozzle_pressure: float = 300000,
                                mesh_coefficient: float = 0.49,
                                 testdata_target_pressure: Optional[float] = 3.0,
                                 pressure_source: str = "linear_fit",
                                 use_testdata_pressure: Optional[bool] = None,
                                 linear_fit_slope: float = TARGET_TO_VALVE_SLOPE,
                                 linear_fit_intercept: float = TARGET_TO_VALVE_INTERCEPT,
                                 ax: Optional[plt.Axes] = None,
                                save_path: Optional[str] = None,
                                show: bool = True,
                                **fan_kwargs) -> Tuple[pd.DataFrame, plt.Figure]:
    """Plot testdata forces and simulated forces on shared simulated y offsets.

    Simulated forces are computed with NozzleForceFan2.calc_force_fan at the
    given y_values (default -10..-70 in steps of 10); their y coordinates are
    kept as-is.

    The nozzle pressure for each simulated point is selected via
    ``pressure_source`` (default ``"linear_fit"``):

    - ``"linear_fit"``: ``P_valve = slope * P_target + intercept`` (bar),
      with ``P_target`` = ``testdata_target_pressure`` when set, else the
      per-file mean target pressure (rank-interpolated like the forces).
      This is the default; it converts the testdata target regulator
      pressure into a nozzle pressure instead of reusing the measured
      valve pressure.
    - ``"testdata"``: the max "average pressure before valve" (bar, x1e5 to
      Pa) of the rank-matched testdata file.
    - ``"fixed"``: the fixed ``nozzle_pressure`` argument for all points.

    For backwards compatibility, the legacy ``use_testdata_pressure`` bool
    (``True`` -> ``"testdata"``, ``False`` -> ``"fixed"``) still overrides
    ``pressure_source`` when explicitly passed.

    Testdata forces are loaded from overview CSVs (one per y position, see
    extract_overview_data). Each file is reduced to a single mean force value
    (optionally filtered to testdata_target_pressure), sorted by its original
    mean stepper y offset, then auto-indexed onto the simulated y grid by rank:
    the testdata point with the lowest original y is plotted at the lowest
    simulated y offset, the second-lowest at the second-lowest, and so on.
    The original CSV y values are NOT used as plot coordinates. If the counts
    differ, testdata ranks are linearly interpolated onto the simulated grid
    (forces, stds, original y, pressures AND target pressures).

    Args:
        testdata_sources: list of overview CSV paths, a glob pattern, or a
            base directory (searched recursively for overview_*.csv).
        testdata_target_pressure: only use rows with this target regulator
            pressure (exact match after float conversion). If None, use all rows.
        pressure_source: ``"linear_fit"`` (default), ``"testdata"`` or ``"fixed"``.
        linear_fit_slope / linear_fit_intercept: fit coefficients [bar/bar, bar].
        Remaining args are forwarded to the simulated sweep.

    Returns (df, fig) where df has columns
        [plot_y_offset, sim_force_n, testdata_force_n, testdata_y_original].
    """
    import glob as _glob
    from pathlib import Path as _Path
    from NozzleForceFan2 import calc_force_fan
    from extract_overview_data import extract_overview_data

    if y_values is None:
        y_values = list(range(-10, -71, -10))
    sim_y = sorted(y_values)

    # --- resolve testdata CSV paths ---
    if isinstance(testdata_sources, (str, _Path)):
        s = str(testdata_sources)
        p = _Path(s)
        if p.is_dir():
            csv_paths = sorted(str(f) for f in p.glob("*/overview_*.csv"))
            if not csv_paths:
                csv_paths = sorted(str(f) for f in p.rglob("overview_*.csv"))
        elif any(ch in s for ch in ("*", "?", "[")):
            csv_paths = sorted(_glob.glob(s, recursive=True))
        else:
            csv_paths = [s]
    else:
        csv_paths = list(testdata_sources)
    if not csv_paths:
        raise ValueError(f"No testdata overview CSVs found in: {testdata_sources}")

    # --- resolve pressure source (legacy bool overrides default) ---
    if use_testdata_pressure is not None:
        pressure_source = "testdata" if use_testdata_pressure else "fixed"
    if pressure_source not in ("linear_fit", "testdata", "fixed"):
        raise ValueError(
            f"Unknown pressure_source={pressure_source!r}; "
            "expected 'linear_fit', 'testdata' or 'fixed'.")

    # --- one mean force (+ valve pressure + target) per CSV, sorted by original y ---
    test_entries = []  # (y_original, mean_force, std_force, n, path, valve_press_bar, target_bar)
    for cp in csv_paths:
        data = extract_overview_data(cp)  # cols: target, pressure, force, yoffset
        if data.size == 0:
            continue
        if testdata_target_pressure is not None:
            mask = np.isclose(data[:, 0], float(testdata_target_pressure), atol=1e-9)
            if not np.any(mask):
                # fall back to closest available target in this file
                targets = np.unique(data[:, 0])
                closest = targets[int(np.argmin(np.abs(targets - testdata_target_pressure)))]
                mask = np.isclose(data[:, 0], closest, atol=1e-9)
            data = data[mask]
            if data.size == 0:
                continue
        y_orig = float(np.mean(data[:, 3]))
        f_mean = float(np.mean(data[:, 2]))
        f_std = float(np.std(data[:, 2], ddof=1)) if len(data) > 1 else 0.0
        p_bar = float(np.max(data[:, 1]))  # max valve pressure for 'testdata' source
        t_bar = float(np.mean(data[:, 0]))  # mean target for 'linear_fit' source
        test_entries.append((y_orig, f_mean, f_std, len(data), str(cp), p_bar, t_bar))
    if not test_entries:
        raise ValueError("No usable testdata rows after filtering.")
    test_entries.sort(key=lambda e: e[0])

    # --- auto-index testdata onto simulated y grid by rank ---
    n_sim, n_test = len(sim_y), len(test_entries)
    test_y_orig = np.array([e[0] for e in test_entries])
    test_f = np.array([e[1] for e in test_entries])
    test_f_std = np.array([e[2] for e in test_entries])
    test_p_bar = np.array([e[5] for e in test_entries])
    test_t_bar = np.array([e[6] for e in test_entries])
    if n_test == n_sim:
        plot_y_test = np.array(sim_y, dtype=float)
        interp_p_bar = test_p_bar.copy()
        interp_t_bar = test_t_bar.copy()
    else:
        # linear rank mapping: test rank r in [0, n_test-1] -> position in sim grid
        src = np.linspace(0, 1, n_test)
        dst = np.linspace(0, 1, n_sim)
        # interpolate test forces onto sim grid points for a comparable line
        test_f = np.interp(dst, src, test_f)
        test_f_std = np.interp(dst, src, test_f_std)
        test_y_orig = np.interp(dst, src, test_y_orig)
        interp_p_bar = np.interp(dst, src, test_p_bar)
        interp_t_bar = np.interp(dst, src, test_t_bar)
        plot_y_test = np.array(sim_y, dtype=float)
    if pressure_source == "linear_fit":
        if testdata_target_pressure is not None:
            fit_targets = np.full(len(sim_y), float(testdata_target_pressure))
        else:
            fit_targets = interp_t_bar
        sim_p_bar = np.array([target_to_valve_pressure_bar(t, linear_fit_slope,
                                                            linear_fit_intercept)
                              for t in fit_targets], dtype=float)
        sim_p_pa = sim_p_bar * 1e5  # bar -> Pa (nozzle_pressure units)
    elif pressure_source == "testdata":
        sim_p_bar = interp_p_bar
        sim_p_pa = sim_p_bar * 1e5  # bar -> Pa (nozzle_pressure units)
        fit_targets = interp_t_bar
    else:  # "fixed"
        sim_p_pa = np.full(len(sim_y), float(nozzle_pressure))
        sim_p_bar = sim_p_pa / 1e5
        fit_targets = np.full(len(sim_y), np.nan)

    # --- simulated sweep (keep y coordinates, pressure from pressure_source) ---
    fan_kwargs.setdefault("print_results", False)
    fan_kwargs.setdefault("graph", False)
    fan_kwargs.setdefault("use_gui", False)
    fan_kwargs.setdefault("mesh_coefficient", mesh_coefficient)
    sim_forces = []
    for y, p_pa in zip(sim_y, sim_p_pa):
        p_use = float(p_pa) if np.isfinite(p_pa) and p_pa > 0 else float(nozzle_pressure)
        total_force, *_ = calc_force_fan(
            name_obj, [x, y, z], cw, nozzle_distance, nozzle_diameter,
            p_use, **fan_kwargs)
        sim_forces.append(float(total_force))

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    ax.errorbar(plot_y_test, test_f, yerr=test_f_std, fmt="s-",
                linewidth=1.5, markersize=6, capsize=4, label="Testdata (rank-mapped)")
    if pressure_source == "linear_fit":
        mean_fit_bar = float(np.mean(sim_p_bar))
        if testdata_target_pressure is not None:
            press_note = (f"sim press.=fit {mean_fit_bar:.3f} bar "
                          f"(target={testdata_target_pressure:g} bar)")
        else:
            press_note = (f"sim press.=fit {mean_fit_bar:.3f} bar mean "
                          f"({linear_fit_slope:.4f}*target+{linear_fit_intercept:.4f})")
        sim_label = (f"Simulated ({name_obj}, fit {mean_fit_bar:.3f} bar "
                     f"from target)")
    elif pressure_source == "testdata":
        sim_label = f"Simulated ({name_obj}, valve press. from testdata)"
        press_note = "sim press.=testdata valve press."
    else:
        sim_label = f"Simulated ({name_obj}, fixed {float(np.mean(sim_p_bar)):.3f} bar)"
        press_note = f"sim press.=fixed {float(np.mean(sim_p_bar)):.3f} bar"
    ax.plot(sim_y, sim_forces, "o-", linewidth=1.5, markersize=6, label=sim_label)
    ax.set_xlabel("Y offset [mm] (simulated grid; testdata auto-indexed by rank)")
    ax.set_ylabel("Force [N]")
    ax.set_title(f"Testdata vs simulated force vs y offset\n"
                 f"(mesh_coeff={mesh_coefficient}, {press_note})")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200)

    if show:
        plt.show()
    else:
        plt.close(fig)

    df = pd.DataFrame({
        "plot_y_offset": np.array(sim_y, dtype=float),
        "sim_target_bar": np.array(fit_targets, dtype=float),
        "sim_pressure_pa": np.array(sim_p_pa, dtype=float),
        "sim_pressure_bar": np.array(sim_p_bar, dtype=float),
        "sim_force_n": np.array(sim_forces, dtype=float),
        "testdata_force_n": np.array(test_f, dtype=float),
        "testdata_force_std_n": np.array(test_f_std, dtype=float),
        "testdata_y_original": np.array(test_y_orig, dtype=float),
    })
    return df, fig


def _cli(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Plot averaged force vs pressure-before-valve from a testrun CSV, or simulated force vs y offset.")
    p.add_argument("csv", nargs="?", default=None, help="Path to the testrun CSV file (not needed with --simulate)")
    p.add_argument("--out", "-o", help="Output image path (optional). If omitted the plot will be shown but not saved.")
    p.add_argument("--no-show", action="store_true", help="Do not call plt.show(); useful when running headless")
    p.add_argument("--simulate", action="store_true",
                   help="Generate force values with NozzleForceFan2.calc_force_fan "
                        "sweeping y from --y-start to --y-stop and plot force vs y offset.")
    p.add_argument("--x", type=float, default=-25.2)
    p.add_argument("--z", type=float, default=0.0)
    p.add_argument("--y-start", type=int, default=-10)
    p.add_argument("--y-stop", type=int, default=-70)
    p.add_argument("--y-step", type=int, default=-10)
    p.add_argument("--pressure", type=float, default=300000,
                   help="Nozzle pressure in Pa for --simulate; in --compare mode only "
                        "used with --pressure-source fixed (default fit from target).")
    p.add_argument("--pressure-source", choices=["linear_fit", "testdata", "fixed"],
                   default="linear_fit",
                   help="In --compare mode, where sim pressure comes from "
                        "(default: linear_fit target->valve).")
    p.add_argument("--fixed-sim-pressure", action="store_true",
                   help="Legacy alias for --pressure-source fixed (uses --pressure).")
    p.add_argument("--testdata-pressure", action="store_true",
                   help="Legacy alias for --pressure-source testdata "
                        "(measured valve pressure).")
    p.add_argument("--fit-slope", type=float, default=TARGET_TO_VALVE_SLOPE,
                   help="Linear-fit slope P_valve=slope*P_target+intercept [bar/bar].")
    p.add_argument("--fit-intercept", type=float, default=TARGET_TO_VALVE_INTERCEPT,
                   help="Linear-fit intercept [bar].")
    p.add_argument("--rays", type=int, default=None,
                   help="ray_number forwarded to calc_force_fan (default: script default).")
    p.add_argument("--mesh-coefficient", type=float, default=0.49,
                   help="mesh coefficient applied to simulated forces (default: 0.49).")
    p.add_argument("--compare", default=None,
                   help="Testdata source for combined plot: base dir, glob, or "
                        "comma-separated overview CSVs. Plots testdata (rank-mapped) "
                        "together with simulated forces.")
    p.add_argument("--target-pressure", type=float, default=3.0,
                   help="Target regulator pressure used to filter testdata "
                        "(default: 3.0). Use 'nan' to disable filtering.")
    args = p.parse_args(argv)

    if args.compare:
        y_values = list(range(args.y_start, args.y_stop - 1, args.y_step))
        fan_kwargs: dict = {}
        if args.rays is not None:
            fan_kwargs["ray_number"] = args.rays
        tp = args.target_pressure
        try:
            tp_val = float("nan") if str(tp).lower() == "nan" else float(tp)
        except (TypeError, ValueError):
            tp_val = 3.0
        target = None if (isinstance(tp_val, float) and np.isnan(tp_val)) else tp_val
        src = [s.strip() for s in str(args.compare).split(",") if s.strip()]
        src_arg = src[0] if len(src) == 1 else src
        source = args.pressure_source
        if args.testdata_pressure:
            source = "testdata"
        if args.fixed_sim_pressure:
            source = "fixed"
        try:
            df, fig = graph_testdata_vs_simulated(
                src_arg, y_values, x=args.x, z=args.z,
                nozzle_pressure=args.pressure, mesh_coefficient=args.mesh_coefficient,
                testdata_target_pressure=target,
                pressure_source=source,
                linear_fit_slope=args.fit_slope,
                linear_fit_intercept=args.fit_intercept,
                save_path=args.out, show=not args.no_show, **fan_kwargs)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 2
        print(df.to_string(index=False))
        if args.out:
            print(f"Saved plot to {args.out}")
        return 0

    if args.simulate:
        y_values = list(range(args.y_start, args.y_stop - 1, args.y_step))
        fan_kwargs: dict = {}
        if args.rays is not None:
            fan_kwargs["ray_number"] = args.rays
        try:
            df, fig = graph_simulated_force_vs_yoffset(
                y_values, x=args.x, z=args.z, nozzle_pressure=args.pressure,
                mesh_coefficient=args.mesh_coefficient,
                save_path=args.out, show=not args.no_show, **fan_kwargs)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 2
        print(df.to_string(index=False))
        if args.out:
            print(f"Saved plot to {args.out}")
        return 0

    if not args.csv:
        print("Error: csv path required unless --simulate is given.", file=sys.stderr)
        return 2
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
