import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

class Conductor:
    """Represents a rectangular conductor."""

    def __init__(
        self,
        voltage: float,
        x_range: Tuple[float,float],
        y_range: Tuple[float,float]
    ) -> None:
        """
        Parameters
        ----------
        voltage :
            Electric potential in Volts.
        x_range :
            `(min_x, max_x)` of the conductor in real coordinates.
        y_range :
            `(min_y, max_y)` of the conductor in real coordinates.

        If the values in the ranges are not multiples of `dx`
        the actual conductor may be smaller.
        """
        self.voltage = voltage
        self.x1, self.x2 = x_range
        self.y1, self.y2 = y_range

class ElectricField:

    def __init__(
        self,
        grid_size: int = 10,
        dx: float = 1e-1
    ) -> None:
        """
        Parameters
        ----------
        grid_size :
            Ex and Ey will have shape `(16*grid_size,9*grid_size)`.
        dx :
            Length in meters of the width (and height) of a cell.
        """
        self.nx = 16*grid_size
        self.ny = 9*grid_size
        self.dy, self.dx = dx, dx

        self.V = np.zeros((self.nx+2, self.ny+2))
        self.conductor_mask = np.zeros((self.nx, self.ny), dtype=bool)

        self.Ex = np.zeros((self.nx, self.ny))
        self.Ey = np.zeros((self.nx, self.ny))

    def add(self, c: Conductor) -> None:
        """
        Add a Conductor to the simulation.
        """
        x1, x2 = max(0, int(c.x1 / self.dx)), min(int(c.x2 / self.dx), self.nx)
        y1, y2 = max(0, int(c.y1 / self.dy)), min(int(c.y2 / self.dy), self.ny)

        self.conductor_mask[x1:x2, y1:y2] = True
        self.V[x1+1:x2+1, y1+1:y2+1] = c.voltage

    def compute(self, it: int = 100000, eps: float = 1e-6) -> None:
        """
        Solve the Laplace equation with Dirichlet boundry condition
        using the Finite Difference Method to compute the electric potential.

        Then use the solution to compute the electric field.
        """
        for i in range(it):
            V_old = self.V.copy()

            self.V[1:-1,1:-1] = np.where(
                self.conductor_mask,
                self.V[1:-1,1:-1],
                (self.V[2:,1:-1] + self.V[:-2,1:-1]
                 + self.V[1:-1,2:] + self.V[1:-1,:-2]) / 4
            )

            delta = np.max(np.abs(self.V - V_old))
            if delta < eps:
                logger.info(f"Converged in {i} iterations")
                break

        self.Ex = np.where(
            self.conductor_mask,
            0,
            -(self.V[2:, 1:-1] - self.V[:-2, 1:-1]) / (2 * self.dx)
        )
        self.Ey = np.where(
            self.conductor_mask,
            0,
            -(self.V[1:-1, 2:] - self.V[1:-1, :-2]) / (2 * self.dy)
        )

    def get_potential_at_batch(
        self,
        x_arr: np.ndarray,
        y_arr: np.ndarray
    ) -> np.ndarray:
        """
        Get electric potential for a batch of points using bilinear interpolation.

        If `(x,y)` is outside simulation this returns
        the value for the closest simulated point.
        """
        i_arr = np.clip(x_arr / self.dx, 0, self.nx - 1)
        j_arr = np.clip(y_arr / self.dy, 0, self.ny - 1)

        i0 = i_arr.astype(int)
        j0 = j_arr.astype(int)
        i1 = np.minimum(i0 + 1, self.nx - 1)
        j1 = np.minimum(j0 + 1, self.ny - 1)

        dx_frac = i_arr - i0
        dy_frac = j_arr - j0

        # Interior V starts at [1,1] due to ghost-cell padding
        V00 = self.V[i0 + 1, j0 + 1]
        V10 = self.V[i1 + 1, j0 + 1]
        V01 = self.V[i0 + 1, j1 + 1]
        V11 = self.V[i1 + 1, j1 + 1]

        return (V00 * (1 - dx_frac) * (1 - dy_frac)
                + V10 * dx_frac       * (1 - dy_frac)
                + V01 * (1 - dx_frac) * dy_frac
                + V11 * dx_frac       * dy_frac)

    def get_field_at(self, x: float, y: float) -> Tuple[float, float]:
        """
        Get electric field vector at point `(x,y)` in real coordinates.

        Uses bilinear interpolation to get values for all real points.

        If `(x,y)` is outside simulation this returns
        the value for the closest simulated point.
        """
        i = max(0.0, min(x / self.dx, self.nx-2))
        j = max(0.0, min(y / self.dy, self.ny-2))

        i0, j0 = int(i), int(j)
        i1, j1 = i0 + 1, j0 + 1

        dx_frac = i - i0
        dy_frac = j - j0

        w_x = np.array([1 - dx_frac, dx_frac])
        w_y = np.array([1 - dy_frac, dy_frac])

        Ex_mat = np.array([[self.Ex[i0, j0], self.Ex[i0, j1]],
                      [self.Ex[i1, j0], self.Ex[i1, j1]]])
        Ey_mat = np.array([[self.Ey[i0, j0], self.Ey[i0, j1]],
                      [self.Ey[i1, j0], self.Ey[i1, j1]]])

        return (
            float(w_x @ Ex_mat @ w_y),
            float(w_x @ Ey_mat @ w_y)
        )

    def get_field_at_batch(
        self,
        x_arr: np.ndarray,
        y_arr:np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get electric field vectors for batch of points in real coordinates.

        Uses bilinear interpolation to get values for all real points.

        If `(x,y)` is outside simulation this returns
        the value for the closest simulated point.
        """
        i_arr = np.clip(x_arr / self.dx, 0, self.nx-2)
        j_arr = np.clip(y_arr / self.dy, 0, self.ny-2)

        i0 = i_arr.astype(int)
        j0 = j_arr.astype(int)
        i1 = i0 + 1
        j1 = j0 + 1

        dx_frac = i_arr - i0
        dy_frac = j_arr - j0

        w_x = np.stack([1 - dx_frac, dx_frac], axis=-1)[:, np.newaxis, :]
        w_y = np.stack([1 - dy_frac, dy_frac], axis=-1)[:, :, np.newaxis]

        Ex_mat = np.stack([
            np.stack([self.Ex[i0, j0], self.Ex[i0, j1]], axis=-1),
            np.stack([self.Ex[i1, j0], self.Ex[i1, j1]], axis=-1)
        ], axis=-2)

        Ey_mat = np.stack([
            np.stack([self.Ey[i0, j0], self.Ey[i0, j1]], axis=-1),
            np.stack([self.Ey[i1, j0], self.Ey[i1, j1]], axis=-1)
        ], axis=-2)

        return (
            (w_x @ Ex_mat @ w_y).flatten(),
            (w_x @ Ey_mat @ w_y).flatten()
        )

class IonGenerator:
    """Represents a rectangular area where ions randomly appear."""

    def __init__(
        self,
        mass: float,
        charge: float,
        init_v: Tuple[float,float],
        x_range: Tuple[float,float],
        y_range: Tuple[float,float],
        *,
        randomize_v: bool = False,
        seed: int | None = None
    ) -> None:
        """
        Parameters
        ----------
        mass :
            Mass of the generated ions in kg.
        charge :
            Charge of the generated ions in C.
        init_v:
            Initial velocity vector in m/s.
        x_range :
            `(min_x, max_x)` of the generation area in real coordinates.
        y_range :
            `(min_y, max_y)` of the generation area in real coordinates.
        randomize_v :
            Whether to randomize initial velocities of the ions.
            If true `init_v` must not be `(0,0)`.
        seed :
            Seed for the random number generator. `None` gives fresh randomness.
        """
        self.mass = mass
        self.charge = charge
        self.init_vx, self.init_vy = init_v
        self.x1, self.x2 = x_range
        self.y1, self.y2 = y_range
        self.randomize_v = randomize_v
        self.seed = seed

    def generate_initial(
        self,
        n: int
    ) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
        """
        Generate the initial real coordinates and velocities of n ions.
        """
        rng = np.random.default_rng(self.seed)

        xs = rng.random((n,)) * (self.x2 - self.x1) + self.x1
        ys = rng.random((n,)) * (self.y2 - self.y1) + self.y1

        if self.randomize_v:
            vxs = (1 + (rng.random((n,)) - 0.5) * 0.05) * self.init_vx
            vys = (1 + (rng.random((n,)) - 0.5) * 0.05) * self.init_vy
        else:
            vxs = np.full((n,), self.init_vx)
            vys = np.full((n,), self.init_vy)

        return (xs, ys, vxs, vys)

class IonSimulation:

    def __init__(
        self,
        field: ElectricField
    ) -> None:
        """
        Parameters
        ----------
        field :
            The computed electric field.
        generator:
            The generator to create the simulated ions.
        n :
            Amount of ions to generate.
        """
        self.field = field
        self._conserve_energy = False

    def step(self, step: int, dt: float) -> None:
        """One step of the simulation."""
        Ex, Ey = self.field.get_field_at_batch(self.x_pos, self.y_pos)
        ax = self.generator.charge * Ex / self.generator.mass
        ay = self.generator.charge * Ey / self.generator.mass

        self.x_vel = self.x_vel + ax * dt
        self.y_vel = self.y_vel + ay * dt

        self.x_pos = self.x_pos + self.x_vel * dt
        self.y_pos = self.y_pos + self.y_vel * dt

        if self._conserve_energy:
            # Rescale |v| so KE + qV stays constant (Farnell et al. 2003).
            V_now = self.field.get_potential_at_batch(self.x_pos, self.y_pos)
            ke_target = np.maximum(
                self._E_total - self.generator.charge * V_now, 0.0
            )
            v_sq = self.x_vel**2 + self.y_vel**2
            v_sq = np.where(v_sq > 0, v_sq, 1.0)
            scale = np.sqrt(ke_target / (0.5 * self.generator.mass * v_sq))
            self.x_vel *= scale
            self.y_vel *= scale

        self.trajectories[:,0,step] = self.x_pos
        self.trajectories[:,1,step] = self.y_pos

    def simulate(
        self,
        generator: IonGenerator,
        n: int,
        steps: int = 20,
        dt: float = 1e-6,
        conserve_energy: bool = False
    ) -> None:
        """
        Parameters
        ----------
        steps :
            Number of steps to simulate.
        dt :
            Delta t in seconds.
        conserve_energy :
            If True, rescale the velocity vector at each step so that total
            mechanical energy (KE + qV) is held constant (Farnell et al. 2003).
        """
        self.generator = generator
        self.n = n
        self.trajectories = np.zeros((self.n, 2, steps+1))
        self._conserve_energy = conserve_energy

        self.x_pos, self.y_pos, self.x_vel, self.y_vel = \
            self.generator.generate_initial(self.n)

        self.trajectories[:,0,0] = self.x_pos
        self.trajectories[:,1,0] = self.y_pos

        if conserve_energy:
            V0 = self.field.get_potential_at_batch(self.x_pos, self.y_pos)
            self._E_total = (
                0.5 * generator.mass * (self.x_vel**2 + self.y_vel**2)
                + generator.charge * V0
            )

        for step in range(1, steps+1):
            self.step(step, dt)

        logger.info(f"Simulated {n} ions for {steps} steps")
