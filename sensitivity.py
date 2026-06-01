"""
It takes the validated NSTAR geometry as a baseline and sweeps each of the four
parameters listed on slide 4 of the presentation one at a time, re-solving the
electric field and re-integrating the Xe+ ions for every value:

    1. Aperture diameter ratio   da / ds   (accel aperture; screen fixed)
    2. Accel grid thickness      ta
    3. Grid gap distance         gap
    4. Voltage ratio             V_screen / |V_accel|   (accel voltage varied)

For each value it records output metrics grouped by the three things the
central question asks about:

    acceleration -> mean exit speed, effective Delta V
    trajectory   -> mean beam divergence half-angle
    efficiency   -> beam transparency (passed ions) and accel-grid impingement
"""

from dataclasses import replace

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10,
})

from benchmark import (
    NSTAR, build_field, run_ions, classify_ions, _velocity_at_exit,
)
from simulation import XENON_CHARGE, XENON_MASS


# =============================================================================
# Single-point metric evaluation (reuses the benchmark pipeline)
# =============================================================================

def compute_metrics(c) -> dict:
    """
    Build the field, run the ions and return a dict of scalar output metrics
    for one Case.  Mirrors benchmark.py's metric definitions, but returns
    numbers
    """
    field = build_field(c)
    sim = run_ions(c, field)
    hits_screen, hits_accel, passes = classify_ions(c, sim, field)

    # ---- acceleration: exit speed + effective Delta V over the beam ions ----
    v_list, dv_list, ang_list = [], [], []
    for ion_idx in passes:
        res = _velocity_at_exit(c, sim, ion_idx)
        if res is None:
            continue
        vx, vy = res
        if vx <= 0:
            continue
        v_list.append(np.hypot(vx, vy))
        ang_list.append(np.degrees(np.arctan2(abs(vy), vx)))

        x0 = sim.trajectories[ion_idx, 0, 0]
        y0 = sim.trajectories[ion_idx, 1, 0]
        x_traj = sim.trajectories[ion_idx, 0, :]
        y_traj = sim.trajectories[ion_idx, 1, :]
        s = np.where(x_traj >= c.x_exit)[0][0]
        i0 = int(np.clip(x0 / c.dx, 0, c.nx - 1)) + 1
        j0 = int(np.clip(y0 / c.dx, 0, c.ny - 1)) + 1
        ie = int(np.clip(x_traj[s] / c.dx, 0, c.nx - 1)) + 1
        je = int(np.clip(y_traj[s] / c.dx, 0, c.ny - 1)) + 1
        dv_list.append(abs(field.V[i0, j0] - field.V[ie, je]))

    n_beam = len(v_list)
    v_arr = np.array(v_list) if v_list else np.array([np.nan])
    ang_arr = np.array(ang_list) if ang_list else np.array([np.nan])
    dv_arr = np.array(dv_list) if dv_list else np.array([np.nan])

    return {
        "v_exit_kms":    v_arr.mean() / 1e3,
        "delta_v_eff":   dv_arr.mean(),
        "divergence":    ang_arr.mean(),
        "eta_div":       float(np.cos(np.radians(ang_arr.mean())) ** 2),
        "n_beam":        n_beam,
        "n_accel_hit":   len(hits_accel),
        "n_screen_hit":  len(hits_screen),
        "transparency":  n_beam / c.n_ions,
    }


# =============================================================================
# Sweep definitions
# =============================================================================

def _set_da_ratio(r):       
    return replace(NSTAR, da=r * NSTAR.ds)

def _set_ta(ta_mm):
    return replace(NSTAR, ta=ta_mm * 1e-3)

def _set_gap(gap_mm):
    return replace(NSTAR, gap=gap_mm * 1e-3)

def _set_voltage_ratio(R):  # R = V_screen / |V_accel|
    return replace(NSTAR, v_accel=-NSTAR.v_screen / R)


SWEEPS = [
    dict(
        key="aperture",
        title="Aperture diameter ratio  da / ds",
        xlabel="Accel / screen aperture ratio  da / ds",
        values=np.round(np.linspace(0.40, 0.90, 8), 3),
        make=_set_da_ratio,
        baseline=NSTAR.da / NSTAR.ds,
    ),
    dict(
        key="thickness",
        title="Accel grid thickness  ta",
        xlabel="Accel grid thickness  ta (mm)",
        values=np.round(np.linspace(0.25, 1.00, 8), 3),
        make=_set_ta,
        baseline=NSTAR.ta * 1e3,
    ),
    dict(
        key="gap",
        title="Screen-to-accel gap",
        xlabel="Grid gap (mm)",
        values=np.round(np.linspace(0.30, 1.20, 8), 3),
        make=_set_gap,
        baseline=NSTAR.gap * 1e3,
    ),
    dict(
        key="voltage",
        title="Voltage ratio  V_screen / |V_accel|",
        xlabel="Voltage ratio  V_screen / |V_accel|",
        values=np.round(np.linspace(2.0, 18.0, 8), 2),
        make=_set_voltage_ratio,
        baseline=NSTAR.v_screen / abs(NSTAR.v_accel),
    ),
]


# =============================================================================
# Run one sweep + plot
# =============================================================================

def run_sweep(spec) -> dict:
    print("\n" + "=" * 65)
    print(f" SWEEP: {spec['title']}")
    print("=" * 65)
    print(f"   baseline NSTAR value = {spec['baseline']:.3f}")
    print(f"   {'x':>8} | {'v_exit':>8} | {'dV_eff':>8} | {'diverg':>7} | "
          f"{'beam':>5} | {'accel':>5}")
    print(f"   {'':>8} | {'km/s':>8} | {'V':>8} | {'deg':>7} | "
          f"{'/50':>5} | {'hit':>5}")

    rows = {k: [] for k in
            ("v_exit_kms", "delta_v_eff", "divergence",
             "eta_div", "n_beam", "n_accel_hit", "transparency")}

    for x in spec["values"]:
        c = spec["make"](x)
        m = compute_metrics(c)
        for k in rows:
            rows[k].append(m[k])
        print(f"   {x:>8.3f} | {m['v_exit_kms']:>8.2f} | "
              f"{m['delta_v_eff']:>8.1f} | {m['divergence']:>7.2f} | "
              f"{m['n_beam']:>5d} | {m['n_accel_hit']:>5d}")

    rows = {k: np.array(v) for k, v in rows.items()}
    _plot_sweep(spec, rows)
    return rows


def _plot_sweep(spec, rows):
    x = spec["values"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(f"Parameter sweep - {spec['title']}  (NSTAR baseline)",
                 fontsize=13)

    def panel(ax, y, ylabel, color, title):
        ax.plot(x, y, "o-", color=color, lw=2)
        ax.axvline(spec["baseline"], ls="--", color="gray", alpha=0.7,
                   label="NSTAR baseline")
        ax.set_xlabel(spec["xlabel"])
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    panel(axes[0, 0], rows["v_exit_kms"], "Mean exit speed (km/s)",
          "tab:blue", "Acceleration - exit speed")
    panel(axes[0, 1], rows["divergence"], "Mean divergence half-angle (deg)",
          "tab:green", "Trajectory - beam divergence")
    panel(axes[1, 0], rows["transparency"] * 100, "Beam transparency (%)",
          "tab:purple", "Efficiency - ions reaching exit")
    panel(axes[1, 1], rows["n_accel_hit"], "Accel-grid impingement (/50)",
          "tab:red", "Efficiency - accel-grid losses")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fname = f"sensitivity_{spec['key']}.png"
    fig.savefig(fname, dpi=150)
    fig.savefig(f"report/figures/sensitivity_{spec['key']}.pdf")
    plt.close(fig)
    print(f"   -> saved {fname}")


def run_sweep_for_report(spec) -> dict:
    """Run a parameter sweep and return the metric rows dict."""
    rows = {k: [] for k in
            ("v_exit_kms", "delta_v_eff", "divergence",
             "eta_div", "n_beam", "n_accel_hit", "transparency")}
    for x in spec["values"]:
        m = compute_metrics(spec["make"](x))
        for k in rows:
            rows[k].append(m[k])
    return {k: np.array(v) for k, v in rows.items()}


def make_sweep_fig(spec, rows) -> plt.Figure:
    """Four-panel parameter sweep figure."""
    x = spec["values"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(f"Parameter sweep - {spec['title']}  (NSTAR baseline)",
                 fontsize=13)

    def panel(ax, y, ylabel, color, title):
        ax.plot(x, y, "o-", color=color, lw=2)
        ax.axvline(spec["baseline"], ls="--", color="gray", alpha=0.7,
                   label="NSTAR baseline")
        ax.set_xlabel(spec["xlabel"])
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    panel(axes[0, 0], rows["v_exit_kms"],         "Mean exit speed (km/s)",
          "tab:blue",   "Acceleration - exit speed")
    panel(axes[0, 1], rows["divergence"],          "Mean divergence half-angle (deg)",
          "tab:green",  "Trajectory - beam divergence")
    panel(axes[1, 0], rows["transparency"] * 100,  "Beam transparency (%)",
          "tab:purple", "Efficiency - ions reaching exit")
    panel(axes[1, 1], rows["n_accel_hit"],         "Accel-grid impingement (/50)",
          "tab:red",    "Efficiency - accel-grid losses")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def main():
    print("Xe+ mass {:.3e} kg, charge {:.3e} C".format(XENON_MASS, XENON_CHARGE))
    for spec in SWEEPS:
        run_sweep(spec)
    print("\nDone. Plots saved as sensitivity_"
          "{aperture,thickness,gap,voltage}.png")


if __name__ == "__main__":
    main()
