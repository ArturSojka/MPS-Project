"""
Accuracy benchmark - validates the simulation against published gridded ion
thruster geometries.

Two reference cases are run:

  * NSTAR   - Wang et al. (2001), Table 1.  Flight thruster (Deep Space 1).
  * SUNSTAR - Farnell et al. (2003), Fig 5. "Scaled Up NSTAR" grid set, a
              ~1.63x linear scale-up of NSTAR run at much higher voltage.

Metrics (per case)
------------------
1. Axial potential profile  - saddle-point shape vs. reference figures
2. Ion exit velocity        - Euler + energy-correction vs. theoretical value
3. Peak gap electric field  - simulation vs. simple DeltaV/gap estimate
4. Ion trajectory focusing  - screen vs. accel grid impingement counts
5. Beam divergence angle    - theta vs. starting radial position
"""

from dataclasses import dataclass, field as dc_field

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10,
})

from simulation import (
    Conductor, ElectricField, IonGenerator, IonSimulation,
    XENON_CHARGE, XENON_MASS,
)


# =============================================================================
# Case definition
# =============================================================================

@dataclass
class Case:
    """A single reference thruster geometry + operating point to benchmark."""
    name: str
    label: str

    v_screen: float     # V  screen grid voltage (also upstream plasma BC)
    v_accel:  float     # V
    v_ref:    float     # V  voltage used for the theoretical exit-speed calculation

    ds:  float          # m  screen aperture diameter
    da:  float          # m  accel  aperture diameter
    ts:  float          # m  screen grid thickness
    ta:  float          # m  accel  grid thickness
    gap: float          # m  screen-to-accel gap
    l:   float          # m  centre-to-centre hole spacing (= domain height)

    upstream: float     # m  discharge-chamber region before the screen grid
    kTe: float = 5.0    # eV electron temperature (sets the Bohm injection speed)

    grid_size: int = 5
    n_ions: int = 50
    steps:  int = 3000
    dt:     float = 1e-10
    seed:   int = 42

    # derived (filled in __post_init__)
    ny: int = dc_field(init=False)
    nx: int = dc_field(init=False)
    dx: float = dc_field(init=False)
    v_bohm: float = dc_field(init=False)

    def __post_init__(self):
        self.ny = 9  * self.grid_size          # domain height = l
        self.nx = 16 * self.grid_size
        self.dx = self.l / self.ny
        # Bohm velocity: v_B = sqrt(kTe * e / m_Xe)
        self.v_bohm = np.sqrt(self.kTe * XENON_CHARGE / XENON_MASS)

    # Geometry in metres
    @property
    def x_sg_lo(self): return self.upstream
    @property
    def x_sg_hi(self): return self.upstream + self.ts
    @property
    def x_ag_lo(self): return self.x_sg_hi + self.gap
    @property
    def x_ag_hi(self): return self.x_ag_lo + self.ta
    @property
    def y_c(self): return self.l / 2
    @property
    def y_sg_ap_lo(self): return self.y_c - self.ds / 2
    @property
    def y_sg_ap_hi(self): return self.y_c + self.ds / 2
    @property
    def y_ag_ap_lo(self): return self.y_c - self.da / 2
    @property
    def y_ag_ap_hi(self): return self.y_c + self.da / 2
    @property
    def x_exit(self):
        # Near the right boundary (V ~ 0) -> clean DeltaV = V_start - 0.
        return (self.nx - 5) * self.dx


# NSTAR: Wang et al. (2001), Table 1
NSTAR = Case(
    name="nstar", label="NSTAR (Wang et al. 2001)",
    v_screen=1074.0, v_accel=-180.0, v_ref=1074.0,
    ds=1.91e-3, da=1.14e-3, ts=0.38e-3, ta=0.51e-3, gap=0.58e-3, l=2.21e-3,
    upstream=1.00e-3,
)

# SUNSTAR: Farnell et al. (2003), Fig 5
SUNSTAR = Case(
    name="sunstar", label="SUNSTAR (Farnell et al. 2003)",
    v_screen=2880.0, v_accel=-360.0, v_ref=2880.0,
    ds=3.10e-3, da=1.86e-3, ts=0.62e-3, ta=0.83e-3, gap=0.945e-3, l=3.60e-3,
    upstream=1.50e-3,
)

CASES = [NSTAR, SUNSTAR]


# =============================================================================
# Field & particle setup
# =============================================================================

def build_field(c: Case) -> ElectricField:
    field = ElectricField(grid_size=c.grid_size, dx=c.dx)

    field.V[0, :] = c.v_screen

    # Screen grid - two solid strips, aperture ds centred on y_c
    field.add(Conductor(c.v_screen, (c.x_sg_lo, c.x_sg_hi), (0,             c.y_sg_ap_lo)))
    field.add(Conductor(c.v_screen, (c.x_sg_lo, c.x_sg_hi), (c.y_sg_ap_hi,  c.l)))

    # Accel grid - two solid strips, aperture da centred on y_c
    field.add(Conductor(c.v_accel,  (c.x_ag_lo, c.x_ag_hi), (0,             c.y_ag_ap_lo)))
    field.add(Conductor(c.v_accel,  (c.x_ag_lo, c.x_ag_hi), (c.y_ag_ap_hi,  c.l)))

    field.compute()
    return field


def run_ions(c: Case, field: ElectricField) -> IonSimulation:
    gen = IonGenerator(
        mass=XENON_MASS,
        charge=XENON_CHARGE,
        init_v=(c.v_bohm, 0.0),
        x_range=(c.x_sg_lo,             c.x_sg_lo + c.dx),
        y_range=(c.y_sg_ap_lo + c.dx,   c.y_sg_ap_hi - c.dx),
        seed=c.seed,
    )
    sim = IonSimulation(field)
    sim.simulate(gen, c.n_ions, c.steps, c.dt, conserve_energy=True)
    return sim


# =============================================================================
# Impingement classification
# =============================================================================

def classify_ions(c: Case, sim: IonSimulation, field: ElectricField):
    """
    Walk each ion's trajectory; stop at the first step where it enters a
    conductor cell. Ions that never hit a conductor and reach x_exit
    are counted as beam ions (passes).
    """
    hits_screen, hits_accel, passes = [], [], []

    for ion_idx in range(sim.n):
        x_traj = sim.trajectories[ion_idx, 0, :]
        y_traj = sim.trajectories[ion_idx, 1, :]
        impinged = False

        for step in range(len(x_traj)):
            i = int(np.clip(x_traj[step] / c.dx, 0, c.nx - 1))
            j = int(np.clip(y_traj[step] / c.dx, 0, c.ny - 1))

            if field.conductor_mask[i, j]:
                if c.x_sg_lo <= x_traj[step] <= c.x_sg_hi:
                    hits_screen.append(ion_idx)
                else:
                    hits_accel.append(ion_idx)
                impinged = True
                break

        if not impinged and x_traj[-1] >= c.x_exit:
            passes.append(ion_idx)

    return hits_screen, hits_accel, passes


def _velocity_at_exit(c: Case, sim: IonSimulation, ion_idx: int):
    """
    Return (vx, vy) estimated from consecutive positions at the first step
    the ion crosses x_exit.  Returns None if the ion never crosses.
    """
    x_traj = sim.trajectories[ion_idx, 0, :]
    y_traj = sim.trajectories[ion_idx, 1, :]
    crossed = np.where(x_traj >= c.x_exit)[0]
    if len(crossed) == 0 or crossed[0] == 0:
        return None
    s = crossed[0]
    vx = (x_traj[s] - x_traj[s - 1]) / c.dt
    vy = (y_traj[s] - y_traj[s - 1]) / c.dt
    return vx, vy


# =============================================================================
# Metric 1 - Axial potential profile
# =============================================================================

def metric_potential_profile(c: Case, field: ElectricField):
    j_c  = int(c.y_c / c.dx)
    x_mm = np.arange(c.nx) * c.dx * 1e3
    v_ax = field.V[1:c.nx + 1, j_c + 1]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x_mm, v_ax, 'b-', lw=2, label='On-axis potential')
    ax.axhline(c.v_screen, ls='--', color='steelblue', alpha=0.6,
               label=f'V_screen = {c.v_screen:.0f} V')
    ax.axhline(c.v_accel,  ls='--', color='tomato',    alpha=0.6,
               label=f'V_accel  = {c.v_accel:.0f} V')
    ax.axvspan(c.x_sg_lo * 1e3, c.x_sg_hi * 1e3, alpha=0.15, color='steelblue',
               label='Screen grid')
    ax.axvspan(c.x_ag_lo * 1e3, c.x_ag_hi * 1e3, alpha=0.15, color='tomato',
               label='Accel grid')

    ax.set_xlabel('x position (mm)')
    ax.set_ylabel('Electric potential (V)')
    ax.set_title(f'Metric 1 - Axial potential profile - {c.label}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'benchmark_{c.name}_potential.png', dpi=150)
    plt.savefig(f'report/figures/benchmark_{c.name}_potential.pdf')
    plt.close(fig)

    v_min_result = v_ax[int(c.x_sg_lo / c.dx):int(c.x_ag_hi / c.dx) + 5].min()
    print("\n1. Potential profile")
    print(f"   On-axis minimum near accel grid : {v_min_result:.1f} V")
    print(f"   (accel grid voltage is {c.v_accel:.0f} V; on-axis value is higher")
    print( "    because the field spreads through the aperture hole)")


# =============================================================================
# Metric 2 - Ion exit velocity
# =============================================================================

def metric_exit_velocity(c: Case, sim: IonSimulation, field: ElectricField, passes):
    if not passes:
        print("\n2. Exit velocity - no ions reached exit line")
        return

    vx_list, vy_list, V_start_list, V_exit_list = [], [], [], []

    for ion_idx in passes:
        result = _velocity_at_exit(c, sim, ion_idx)
        if result is None:
            continue
        vx, vy = result
        vx_list.append(vx)
        vy_list.append(vy)

        x0 = sim.trajectories[ion_idx, 0, 0]
        y0 = sim.trajectories[ion_idx, 1, 0]
        x_traj = sim.trajectories[ion_idx, 0, :]
        y_traj = sim.trajectories[ion_idx, 1, :]
        s = np.where(x_traj >= c.x_exit)[0][0]

        i0 = int(np.clip(x0 / c.dx, 0, c.nx - 1)) + 1
        j0 = int(np.clip(y0 / c.dx, 0, c.ny - 1)) + 1
        ie = int(np.clip(x_traj[s] / c.dx, 0, c.nx - 1)) + 1
        je = int(np.clip(y_traj[s] / c.dx, 0, c.ny - 1)) + 1
        V_start_list.append(field.V[i0, j0])
        V_exit_list.append(field.V[ie, je])

    if not vx_list:
        print("\n2. Exit velocity - trajectory crossing data unavailable")
        return

    vx_arr = np.array(vx_list)
    vy_arr = np.array(vy_list)
    v_arr  = np.sqrt(vx_arr**2 + vy_arr**2)
    V_s    = np.array(V_start_list)
    V_e    = np.array(V_exit_list)

    v_theory_per_ion = np.sqrt(2 * XENON_CHARGE * np.abs(V_s - V_e) / XENON_MASS)
    error_pct = np.abs(v_arr - v_theory_per_ion) / v_theory_per_ion * 100

    v_ref = np.sqrt(2 * XENON_CHARGE * c.v_ref / XENON_MASS)
    delta_V_actual = np.abs(V_s - V_e).mean()

    print("\n2. Ion exit velocity")
    print(f"   Reference (DeltaV={c.v_ref:.0f} V, Xe+) : {v_ref:>10,.0f} m/s "
          f"({v_ref/1e3:.2f} km/s)")
    print(f"   Simulated mean vx at x_exit     : {vx_arr.mean():>10,.0f} m/s "
          f"({vx_arr.mean()/1e3:.2f} km/s)")
    print(f"   Simulated mean |v| at x_exit    : {v_arr.mean():>10,.0f} m/s "
          f"({v_arr.mean()/1e3:.2f} km/s)")
    print(f"   Mean V_start (ion entry)        : {V_s.mean():.1f} V  "
          f"(expected ~{c.v_ref:.0f} V)")
    print(f"   Mean DeltaV experienced         : {delta_V_actual:.1f} V  "
          f"(expected {c.v_ref:.0f} V)")
    print(f"   Mean energy conservation error  : {error_pct.mean():.2f} %")
    print( "   Note: V_start < V_ref because our 2D Laplace model has no")
    print( "   upstream plasma sheath - the accel grid lowers aperture V.")


# =============================================================================
# Metric 3 - Peak electric field in the inter-grid gap
# =============================================================================

def metric_gap_field(c: Case, field: ElectricField):
    ix_lo = int(c.x_sg_hi / c.dx)
    ix_hi = int(c.x_ag_lo / c.dx)
    j_c   = int(c.y_c / c.dx)

    # On-axis (centreline) peak
    E_center = np.sqrt(field.Ex[ix_lo:ix_hi + 1, j_c]**2
                       + field.Ey[ix_lo:ix_hi + 1, j_c]**2).max()

    E_est = (c.v_screen - c.v_accel) / c.gap   # 1D parallel-plate estimate

    print("\n3. Peak electric field in gap")
    print(f"   Uniform 1D estimate  DeltaV/gap : {E_est/1e6:.2f} MV/m")
    print(f"   Simulation peak |E| (on-axis)   : {E_center/1e6:.2f} MV/m")
    print(f"   Ratio  sim / estimate           : {E_center / E_est:.2f}")
    print( "   (< 1: aperture holes spread field; 1D estimate overestimates)")


# =============================================================================
# Metric 4 - Focusing & impingement
# =============================================================================

def metric_focusing(hits_screen, hits_accel, passes, sim: IonSimulation):
    total = sim.n
    print("\n4. Ion trajectory focusing")
    print(f"   Ions in beam (reached x_exit)   : {len(passes):3d} / {total}")
    print(f"   Impinged on screen grid         : {len(hits_screen):3d} / {total}"
          "  <- should be 0")
    print(f"   Impinged on accel  grid         : {len(hits_accel):3d} / {total}")


# =============================================================================
# Metric 5 - Beam divergence angle
# =============================================================================

def metric_divergence(c: Case, sim: IonSimulation, passes):
    if not passes:
        print("\n5. Divergence - no ions reached exit line")
        return

    vx_list, vy_list, y_init_list = [], [], []

    for ion_idx in passes:
        result = _velocity_at_exit(c, sim, ion_idx)
        if result is None:
            continue
        vx, vy = result
        if vx <= 0:
            continue   # backward-moving ion 
        vx_list.append(vx)
        vy_list.append(vy)
        y_init_list.append(sim.trajectories[ion_idx, 1, 0])

    if not vx_list:
        print("\n5. Divergence - no valid exit velocities")
        return

    vx_arr = np.array(vx_list)
    vy_arr = np.array(vy_list)
    y_init = np.array(y_init_list)

    angles = np.degrees(np.arctan2(np.abs(vy_arr), vx_arr))
    sort   = np.argsort(y_init)
    p95    = np.percentile(angles, 95)
    eta_d  = np.cos(np.radians(angles.mean()))**2

    print("\n5. Beam divergence")
    print(f"   Mean divergence angle           : {angles.mean():.2f} deg")
    print(f"   95th-percentile angle           : {p95:.2f} deg")
    print(f"   Divergence efficiency eta_d     : {eta_d:.4f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(
        (y_init - c.y_c)[sort] * 1e3,
        angles[sort],
        c='steelblue', s=40, zorder=5,
    )
    ax.axhline(angles.mean(), ls='--', color='tomato',
               label=f'Mean = {angles.mean():.2f} deg')
    ax.axhline(p95, ls=':', color='orange',
               label=f'95th pct = {p95:.2f} deg')
    ax.set_xlabel('Starting position relative to centreline (mm)')
    ax.set_ylabel('Exit divergence angle (deg)')
    ax.set_title(f'Metric 5 - Beam divergence vs. radial start - {c.label}')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'benchmark_{c.name}_divergence.png', dpi=150)
    plt.savefig(f'report/figures/benchmark_{c.name}_divergence.pdf')
    plt.close(fig)

    return angles, y_init


# =============================================================================
# Combined overview plot
# =============================================================================

def plot_overview(c: Case, field: ElectricField, sim: IonSimulation,
                  passes, hits_accel, hits_screen):
    fig, ax = plt.subplots(figsize=(13, 5))

    im = ax.imshow(
        field.V[1:c.nx + 1, 1:c.ny + 1].T,
        origin='lower', aspect='auto',
        extent=[0, c.nx * c.dx * 1e3, 0, c.ny * c.dx * 1e3],
        cmap='RdBu_r', vmin=c.v_accel, vmax=c.v_screen,
    )
    plt.colorbar(im, ax=ax, label='Potential (V)', shrink=0.8)

    def draw_conductor(xl, xh, yl, yh, color):
        ax.add_patch(mpatches.Rectangle(
            (xl * 1e3, yl * 1e3), (xh - xl) * 1e3, (yh - yl) * 1e3,
            linewidth=1, edgecolor='k', facecolor=color, alpha=0.5,
        ))

    draw_conductor(c.x_sg_lo, c.x_sg_hi, 0,             c.y_sg_ap_lo, 'steelblue')
    draw_conductor(c.x_sg_lo, c.x_sg_hi, c.y_sg_ap_hi,  c.l,          'steelblue')
    draw_conductor(c.x_ag_lo, c.x_ag_hi, 0,             c.y_ag_ap_lo, 'tomato')
    draw_conductor(c.x_ag_lo, c.x_ag_hi, c.y_ag_ap_hi,  c.l,          'tomato')

    ax.axvline(c.x_exit * 1e3, ls=':', color='white', lw=1, alpha=0.7,
               label='Exit measurement line')

    for ion_idx in passes:
        ax.plot(sim.trajectories[ion_idx, 0, :] * 1e3,
                sim.trajectories[ion_idx, 1, :] * 1e3,
                'lime', lw=0.8, alpha=0.7)
    for ion_idx in hits_accel:
        ax.plot(sim.trajectories[ion_idx, 0, :] * 1e3,
                sim.trajectories[ion_idx, 1, :] * 1e3,
                'r-', lw=0.8, alpha=0.7)
    for ion_idx in hits_screen:
        ax.plot(sim.trajectories[ion_idx, 0, :] * 1e3,
                sim.trajectories[ion_idx, 1, :] * 1e3,
                'm-', lw=0.8, alpha=0.7)

    handles = [
        mpatches.Patch(color='lime',      label='Passed (beam)'),
        mpatches.Patch(color='red',       label='Accel impingement'),
        mpatches.Patch(color='magenta',   label='Screen impingement'),
        mpatches.Patch(color='steelblue', alpha=0.5, label='Screen grid'),
        mpatches.Patch(color='tomato',    alpha=0.5, label='Accel grid'),
    ]
    ax.legend(handles=handles, fontsize=8, loc='upper right')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_title(f'Overview - potential + Xe+ ion trajectories - {c.label}')
    plt.tight_layout()
    plt.savefig(f'benchmark_{c.name}_overview.png', dpi=150)
    plt.savefig(f'report/figures/benchmark_{c.name}_overview.pdf')
    plt.close(fig)


# =============================================================================
# Report API
# =============================================================================

def compute_potential_metrics(c: Case, field: ElectricField) -> dict:
    j_c   = int(c.y_c / c.dx)
    v_ax  = field.V[1:c.nx + 1, j_c + 1]
    v_min = float(v_ax[int(c.x_sg_lo / c.dx):int(c.x_ag_hi / c.dx) + 5].min())
    return {"v_min_axial": v_min}


def compute_gap_field_metrics(c: Case, field: ElectricField) -> dict:
    ix_lo = int(c.x_sg_hi / c.dx)
    ix_hi = int(c.x_ag_lo / c.dx)
    j_c   = int(c.y_c / c.dx)
    E_sim = float(np.sqrt(field.Ex[ix_lo:ix_hi + 1, j_c]**2
                          + field.Ey[ix_lo:ix_hi + 1, j_c]**2).max())
    E_est = (c.v_screen - c.v_accel) / c.gap
    return {"E_sim": E_sim, "E_est": float(E_est), "E_ratio": float(E_sim / E_est)}


def compute_exit_velocity_metrics(c: Case, sim: IonSimulation,
                                  field: ElectricField, passes) -> dict:
    if not passes:
        return {}
    vx_list, vy_list, V_start_list, V_exit_list = [], [], [], []
    for ion_idx in passes:
        result = _velocity_at_exit(c, sim, ion_idx)
        if result is None:
            continue
        vx, vy = result
        vx_list.append(vx)
        vy_list.append(vy)
        x0     = sim.trajectories[ion_idx, 0, 0]
        y0     = sim.trajectories[ion_idx, 1, 0]
        x_traj = sim.trajectories[ion_idx, 0, :]
        y_traj = sim.trajectories[ion_idx, 1, :]
        s  = np.where(x_traj >= c.x_exit)[0][0]
        i0 = int(np.clip(x0       / c.dx, 0, c.nx - 1)) + 1
        j0 = int(np.clip(y0       / c.dx, 0, c.ny - 1)) + 1
        ie = int(np.clip(x_traj[s] / c.dx, 0, c.nx - 1)) + 1
        je = int(np.clip(y_traj[s] / c.dx, 0, c.ny - 1)) + 1
        V_start_list.append(field.V[i0, j0])
        V_exit_list.append(field.V[ie, je])
    if not vx_list:
        return {}
    vx_arr = np.array(vx_list)
    vy_arr = np.array(vy_list)
    v_arr  = np.sqrt(vx_arr**2 + vy_arr**2)
    V_s    = np.array(V_start_list)
    V_e    = np.array(V_exit_list)
    v_theory  = np.sqrt(2 * XENON_CHARGE * np.abs(V_s - V_e) / XENON_MASS)
    error_pct = np.abs(v_arr - v_theory) / v_theory * 100
    return {
        "v_exit_mean":      float(v_arr.mean()),
        "vx_exit_mean":     float(vx_arr.mean()),
        "energy_error_pct": float(error_pct.mean()),
        "delta_v_actual":   float(np.abs(V_s - V_e).mean()),
    }


def compute_divergence_metrics(c: Case, sim: IonSimulation, passes) -> dict:
    if not passes:
        return {}
    vx_list, vy_list, y_init_list = [], [], []
    for ion_idx in passes:
        result = _velocity_at_exit(c, sim, ion_idx)
        if result is None:
            continue
        vx, vy = result
        if vx <= 0:
            continue
        vx_list.append(vx)
        vy_list.append(vy)
        y_init_list.append(sim.trajectories[ion_idx, 1, 0])
    if not vx_list:
        return {}
    vx_arr = np.array(vx_list)
    vy_arr = np.array(vy_list)
    angles = np.degrees(np.arctan2(np.abs(vy_arr), vx_arr))
    return {
        "divergence_mean": float(angles.mean()),
        "divergence_p95":  float(np.percentile(angles, 95)),
        "eta_d":           float(np.cos(np.radians(angles.mean()))**2),
        "_angles":         angles,
        "_y_init":         np.array(y_init_list),
    }


def run_case_for_report(c: Case) -> dict:
    """Run the full benchmark pipeline; return all physics objects and metrics."""
    field = build_field(c)
    sim   = run_ions(c, field)
    hits_screen, hits_accel, passes = classify_ions(c, sim, field)
    results = {
        "field": field, "sim": sim,
        "hits_screen": hits_screen, "hits_accel": hits_accel, "passes": passes,
        "n_ions": c.n_ions,
    }
    results.update(compute_potential_metrics(c, field))
    results.update(compute_gap_field_metrics(c, field))
    results.update(compute_exit_velocity_metrics(c, sim, field, passes))
    results.update(compute_divergence_metrics(c, sim, passes))
    return results


def make_overview_fig(c: Case, r: dict) -> plt.Figure:
    """Potential map with ion trajectories overlaid."""
    field       = r["field"]
    sim         = r["sim"]
    passes      = r["passes"]
    hits_accel  = r["hits_accel"]
    hits_screen = r["hits_screen"]

    fig, ax = plt.subplots(figsize=(13, 5))
    im = ax.imshow(
        field.V[1:c.nx + 1, 1:c.ny + 1].T,
        origin='lower', aspect='auto',
        extent=[0, c.nx * c.dx * 1e3, 0, c.ny * c.dx * 1e3],
        cmap='RdBu_r', vmin=c.v_accel, vmax=c.v_screen,
    )
    plt.colorbar(im, ax=ax, label='Potential (V)', shrink=0.8)

    def _rect(xl, xh, yl, yh, color):
        ax.add_patch(mpatches.Rectangle(
            (xl * 1e3, yl * 1e3), (xh - xl) * 1e3, (yh - yl) * 1e3,
            linewidth=1, edgecolor='k', facecolor=color, alpha=0.5,
        ))

    _rect(c.x_sg_lo, c.x_sg_hi, 0,            c.y_sg_ap_lo, 'steelblue')
    _rect(c.x_sg_lo, c.x_sg_hi, c.y_sg_ap_hi, c.l,          'steelblue')
    _rect(c.x_ag_lo, c.x_ag_hi, 0,            c.y_ag_ap_lo, 'tomato')
    _rect(c.x_ag_lo, c.x_ag_hi, c.y_ag_ap_hi, c.l,          'tomato')
    ax.axvline(c.x_exit * 1e3, ls=':', color='white', lw=1, alpha=0.7)

    for ion_idx in passes:
        ax.plot(sim.trajectories[ion_idx, 0, :] * 1e3,
                sim.trajectories[ion_idx, 1, :] * 1e3, 'lime', lw=0.8, alpha=0.7)
    for ion_idx in hits_accel:
        ax.plot(sim.trajectories[ion_idx, 0, :] * 1e3,
                sim.trajectories[ion_idx, 1, :] * 1e3, 'r-', lw=0.8, alpha=0.7)
    for ion_idx in hits_screen:
        ax.plot(sim.trajectories[ion_idx, 0, :] * 1e3,
                sim.trajectories[ion_idx, 1, :] * 1e3, 'm-', lw=0.8, alpha=0.7)

    handles = [
        mpatches.Patch(color='lime',      label='Passed (beam)'),
        mpatches.Patch(color='red',       label='Accel impingement'),
        mpatches.Patch(color='magenta',   label='Screen impingement'),
        mpatches.Patch(color='steelblue', alpha=0.5, label='Screen grid'),
        mpatches.Patch(color='tomato',    alpha=0.5, label='Accel grid'),
    ]
    ax.legend(handles=handles, fontsize=8, loc='upper right')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_title(f'Overview - potential + Xe+ ion trajectories - {c.label}')
    plt.tight_layout()
    return fig


def make_potential_fig(c: Case, r: dict) -> plt.Figure:
    """On-axis electric potential profile."""
    field = r["field"]
    j_c   = int(c.y_c / c.dx)
    x_mm  = np.arange(c.nx) * c.dx * 1e3
    v_ax  = field.V[1:c.nx + 1, j_c + 1]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x_mm, v_ax, 'b-', lw=2, label='On-axis potential')
    ax.axhline(c.v_screen, ls='--', color='steelblue', alpha=0.6,
               label=f'V_screen = {c.v_screen:.0f} V')
    ax.axhline(c.v_accel,  ls='--', color='tomato',    alpha=0.6,
               label=f'V_accel  = {c.v_accel:.0f} V')
    ax.axvspan(c.x_sg_lo * 1e3, c.x_sg_hi * 1e3, alpha=0.15, color='steelblue',
               label='Screen grid')
    ax.axvspan(c.x_ag_lo * 1e3, c.x_ag_hi * 1e3, alpha=0.15, color='tomato',
               label='Accel grid')
    ax.set_xlabel('x position (mm)')
    ax.set_ylabel('Electric potential (V)')
    ax.set_title(f'On-axis potential profile - {c.label}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def make_divergence_fig(c: Case, r: dict) -> plt.Figure:
    """Exit divergence angle vs. radial starting position."""
    angles   = r["_angles"]
    y_init   = r["_y_init"]
    div_mean = r["divergence_mean"]
    div_p95  = r["divergence_p95"]
    sort     = np.argsort(y_init)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter((y_init - c.y_c)[sort] * 1e3, angles[sort],
               c='steelblue', s=40, zorder=5)
    ax.axhline(div_mean, ls='--', color='tomato',
               label=f'Mean = {div_mean:.2f} deg')
    ax.axhline(div_p95, ls=':', color='orange',
               label=f'95th pct = {div_p95:.2f} deg')
    ax.set_xlabel('Starting position relative to centreline (mm)')
    ax.set_ylabel('Exit divergence angle (deg)')
    ax.set_title(f'Beam divergence vs. radial start - {c.label}')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# =============================================================================
def run_case(c: Case):
    print("\n" + "=" * 65)
    print(f" BENCHMARK CASE: {c.label}")
    print("=" * 65)
    print(f"   grid {c.nx} x {c.ny} cells, dx = {c.dx*1e6:.1f} um, "
          f"domain {c.nx*c.dx*1e3:.2f} x {c.ny*c.dx*1e3:.2f} mm")
    print(f"   V_screen {c.v_screen:.0f} V, V_accel {c.v_accel:.0f} V, "
          f"Bohm v {c.v_bohm:.0f} m/s")

    print("Building geometry and solving electric field ...")
    field = build_field(c)

    print("Running Xe+ ion simulation ...")
    sim = run_ions(c, field)

    print("Classifying ion trajectories ...")
    hits_screen, hits_accel, passes = classify_ions(c, sim, field)

    plot_overview(c, field, sim, passes, hits_accel, hits_screen)
    metric_potential_profile(c, field)
    metric_gap_field(c, field)
    metric_exit_velocity(c, sim, field, passes)
    metric_focusing(hits_screen, hits_accel, passes, sim)
    metric_divergence(c, sim, passes)


def main():
    for c in CASES:
        run_case(c)
    print("\nDone. Plots saved as benchmark_<case>_{overview,potential,divergence}.png")


if __name__ == "__main__":
    main()
