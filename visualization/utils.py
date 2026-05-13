import numpy as np
from typing import List, Tuple, overload


class CoordsConverter:
    """Convert between three 2D coordinate systems:

    - array  : integer indices in the simulation array
    - real   : physical simulation coordinates
    - manim  : manim scene coordinates, origin at center
    """

    def __init__(
        self,
        arr_max_x: int, arr_max_y: int,
        real_max_x: float, real_max_y: float,
        dx: float,
        manim_width: float, manim_height: float,
    ) -> None:
        self.amx, self.amy = arr_max_x, arr_max_y
        self.rmx, self.rmy = real_max_x, real_max_y
        self.dx = dx
        self.manim_width, self.manim_height = manim_width, manim_height

    @overload
    def real2array(self, x: float, y: float) -> Tuple[int, int]:
        ...

    @overload
    def real2array(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        ...

    def real2array(
        self, x: float | np.ndarray, y: float | np.ndarray
    ) -> Tuple[int, int] | Tuple[np.ndarray, np.ndarray]:
        if isinstance(x, (float, int)):
            i = min(max(0, int(x / self.dx)), self.amx - 1)
            j = min(max(0, int(y / self.dx)), self.amy - 1)
            return i, j

        i_arr = np.clip(x / self.dx, 0, self.amx - 1).astype(int)
        j_arr = np.clip(y / self.dx, 0, self.amy - 1).astype(int)
        return i_arr, j_arr

    @overload
    def array2real(self, x: int, y: int) -> Tuple[float, float]:
        ...

    @overload
    def array2real(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        ...

    def array2real(
        self, x: int | np.ndarray, y: int | np.ndarray
    ) -> Tuple[float, float] | Tuple[np.ndarray, np.ndarray]:
        return x * self.dx, y * self.dx

    @overload
    def manim2real(
        self, x_or_points: float, y: float
    ) -> Tuple[float, float]:
        ...

    @overload
    def manim2real(
        self, x_or_points: np.ndarray, y: None = None
    ) -> Tuple[float, float]:
        ...

    @overload
    def manim2real(
        self, x_or_points: List[np.ndarray], y: None = None
    ) -> List[Tuple[float, float]]:
        ...

    def manim2real(
        self,
        x_or_points: float | np.ndarray | List[np.ndarray],
        y: float | None = None,
    ) -> Tuple[float, float] | List[Tuple[float, float]]:
        rx = self.rmx / self.manim_width
        ry = self.rmy / self.manim_height

        if isinstance(x_or_points, (float, int)):
            return (
                (x_or_points + self.manim_width / 2) * rx,
                (y           + self.manim_height / 2) * ry,
            )
        if isinstance(x_or_points, np.ndarray):
            return (
                (x_or_points[0] + self.manim_width / 2) * rx,
                (x_or_points[1] + self.manim_height / 2) * ry,
            )

        return [
            ((p[0] + self.manim_width / 2) * rx,
             (p[1] + self.manim_height / 2) * ry)
            for p in x_or_points
        ]

    @overload
    def real2manim(self, x: float, y: float) -> np.ndarray:
        ...

    @overload
    def real2manim(
        self, x: np.ndarray, y: np.ndarray
    ) -> List[np.ndarray]:
        ...

    def real2manim(
        self, x: float | np.ndarray, y: float | np.ndarray
    ) -> np.ndarray | List[np.ndarray]:
        mx = self.manim_width / self.rmx
        my = self.manim_height / self.rmy

        if isinstance(x, (float, int)):
            return np.array([
                (x - self.rmx / 2) * mx,
                (y - self.rmy / 2) * my,
                0.0,
            ])

        return [
            np.array([
                (xp - self.rmx / 2) * mx,
                (yp - self.rmy / 2) * my,
                0.0,
            ])
            for xp, yp in zip(x, y)
        ]

    @overload
    def manim2array(
        self, x_or_points: float, y: float
    ) -> Tuple[int, int]:
        ...

    @overload
    def manim2array(
        self, x_or_points: np.ndarray, y: None = None
    ) -> Tuple[float, float]:
        ...

    @overload
    def manim2array(
        self, x_or_points: List[np.ndarray], y: None = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        ...

    def manim2array(
        self,
        x_or_points: float | np.ndarray | List[np.ndarray],
        y: float | None = None,
    ) -> Tuple[int, int] | Tuple[np.ndarray, np.ndarray]:
        if isinstance(x_or_points, (float, int, np.ndarray)):
            return self.real2array(*self.manim2real(x_or_points, y))

        r = np.array(self.manim2real(x_or_points, y))
        return self.real2array(r[:, 0], r[:, 1])

    @overload
    def array2manim(self, x: int, y: int) -> np.ndarray:
        ...

    @overload
    def array2manim(
        self, x: np.ndarray, y: np.ndarray
    ) -> List[np.ndarray]:
        ...

    def array2manim(
        self, x: int | np.ndarray, y: int | np.ndarray
    ) -> np.ndarray | List[np.ndarray]:
        return self.real2manim(*self.array2real(x, y))
        