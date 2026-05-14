from typing import Iterable, List, Tuple

import numpy as np
from manim import (
    BLACK,
    ORANGE,
    RESAMPLING_ALGORITHMS,
    Animation,
    Dot,
    ImageMobject,
    MoveAlongPath,
    ParsableManimColor,
    Rectangle,
    TracedPath,
    Vector,
    VGroup,
    VMobject,
    color_gradient,
)
from matplotlib import cm, colors

from simulation import Conductor, ElectricField, IonGenerator, IonSimulation

from .utils import CoordsConverter


class MConductor:

    def __init__(
        self,
        conductor: Conductor,
        converter: CoordsConverter,
        color: ParsableManimColor,
        *,
        fill: bool = True,
        fill_opacity: float = 1.0
    ) -> None:
        self.conductor = conductor
        self.converter = converter
        self.color = color
        self.fill = fill
        self.fill_opacity = fill_opacity

    def as_rect(self) -> Rectangle:
        p1 = self.converter.real2manim(self.conductor.x1,self.conductor.y1)
        p2 = self.converter.real2manim(self.conductor.x2,self.conductor.y2)

        rect = Rectangle(
            color=self.color,
            width=p2[0]-p1[0],
            height=p2[1]-p1[1]
        )
        rect.move_to((p1+p2)/2)

        if self.fill:
            rect.set_fill(color=self.color,opacity=self.fill_opacity)

        return rect


class MElectricField:

    def __init__(
        self,
        field: ElectricField,
        converter: CoordsConverter,
        cmap: str | colors.Colormap = 'RdBu_r',
        color: ParsableManimColor = BLACK,
        opacity: float = 0.6,
        max_arrow_len: float = 1.0
    ) -> None:
        self.field = field
        self.converter = converter
        self.cmap = cmap
        self.color = color
        self.opacity = opacity
        self.max_arrow_len = max_arrow_len

    def as_bg_image(self) -> ImageMobject:
        """
        The electric potential field as a background image.

        Should be added to the scene first, before any other objects.
        """
        data = self.field.V[1:-1, 1:-1].T

        norm = colors.Normalize(vmin=np.nanmin(data), vmax=np.nanmax(data))
        scalar_mappable = cm.ScalarMappable(norm=norm, cmap=self.cmap)

        image = ImageMobject(scalar_mappable.to_rgba(data, bytes=True))
        image.set_resampling_algorithm(RESAMPLING_ALGORITHMS['cubic'])
        image.scale_to_fit_width(self.converter.manim_width)
        image.scale_to_fit_height(self.converter.manim_height)

        return image

    def as_quiver(self, arrow_every: int = 1) -> VGroup:
        """
        The electric field as arrow vectors.
        """
        xs, ys = np.meshgrid(
            np.arange(0, self.converter.amx, arrow_every),
            np.arange(0, self.converter.amy, arrow_every)
        )
        xs, ys = self.converter.array2real(xs.flatten(), ys.flatten())
        Ex, Ey = self.field.get_field_at_batch(xs, ys)
        scale = self.max_arrow_len / np.max(np.sqrt(Ex**2 + Ey**2))
        Ex *= scale
        Ey *= scale
        points = self.converter.real2manim(xs, ys)

        quiver = VGroup()
        quiver.add(
            Vector(np.array([ex, ey, 0.0])).shift(p)
            for ex, ey, p in zip(Ex, Ey, points)
        )
        quiver.set_opacity(self.opacity)
        quiver.set_color(self.color)
        return quiver
        

class MIonGenerator:

    def __init__(
        self,
        generator: IonGenerator,
        converter: CoordsConverter,
        color: ParsableManimColor = ORANGE,
        *,
        fill: bool = False,
        fill_opacity: float = 1.0
    ) -> None:
        self.generator = generator
        self.converter = converter
        self.color = color
        self.fill = fill
        self.fill_opacity = fill_opacity

    def as_rect(self) -> Rectangle:
        p1 = self.converter.real2manim(self.generator.x1,self.generator.y1)
        p2 = self.converter.real2manim(self.generator.x2,self.generator.y2)

        rect = Rectangle(
            color=self.color,
            width=p2[0]-p1[0],
            height=p2[1]-p1[1]
        )
        rect.move_to((p1+p2)/2)

        if self.fill:
            rect.set_fill(color=self.color,opacity=self.fill_opacity)

        return rect


class MIonSimulation:
    """
    Represents a simulation of a group of ions.
    Provides multiple visualization methods.

    A visualization method should be called after every `sim.simulate(...)`.
    """

    def __init__(self,sim: IonSimulation, converter: CoordsConverter) -> None:
        self.sim = sim
        self.converter = converter

    def get_ions(
        self,
        gradient: Iterable[ParsableManimColor]
    ) -> List[Tuple[VMobject,Dot]]:
        ions = []
        for i, c in enumerate(color_gradient(gradient, self.sim.n)):
            path = VMobject(stroke_color=c)
            traj = self.converter.real2manim(
                self.sim.trajectories[i,0,:],
                self.sim.trajectories[i,1,:]
            )
            path.set_points_smoothly(traj)

            dot = Dot(traj[0], 0.04, color=c)

            ions.append((path,dot))
        return ions

    def as_paths(
        self,
        gradient: Iterable[ParsableManimColor]
    ) -> List[VMobject]:
        """
        Return just the paths of the ions.
        """
        ions = self.get_ions(gradient)
        return [path for path, _ in ions]

    def as_at_once(
        self,
        gradient: Iterable[ParsableManimColor]
    ) -> Tuple[List[VMobject],List[Animation]]:
        """
        Animate releasing all ions at once.

        Usage
        -----
        ```
        objects, animations = m_sim.as_at_once(...)
        self.add(*objects)
        self.play(*animations,rate_func=linear,run_time=...)
        self.wait()
        ```
        """
        ions = self.get_ions(gradient)

        animations = [
            MoveAlongPath(dot,path)
            for path, dot in ions
        ]
        objects = []
        for _, dot in ions:
            objects.append(dot)

            trace = TracedPath(
                dot.get_center,
                dissipating_time=0.5,
                stroke_width=1.0,
                stroke_color=dot.get_color(),
                stroke_opacity=[0, 1]
            )

            objects.append(trace)

        return objects, animations
