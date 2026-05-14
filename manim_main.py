from manim import *

from simulation import (
    ELECTRON_CHARGE,
    ELECTRON_MASS,
    PROTON_CHARGE,
    PROTON_MASS,
    Conductor,
    ElectricField,
    IonGenerator,
    IonSimulation,
)
from visualization import (
    CoordsConverter,
    MConductor,
    MElectricField,
    MIonGenerator,
    MIonSimulation,
)


class ElectricFieldScene(Scene):
    def construct(self):
        field = ElectricField(10, 0.1)
        conv = CoordsConverter(
            field.nx, field.ny,
            field.nx*field.dx, field.ny*field.dy,
            field.dx, config["frame_width"], config["frame_height"]
        )
        m_field = MElectricField(field, conv)

        cond1 = Conductor(-1, (1.5, 2.5), (4.0, 5.0))
        field.add(cond1)
        m_cond1 = MConductor(cond1, conv, BLUE)

        cond2 = Conductor(1, (13.5, 14.5), (4.0, 5.0))
        field.add(cond2)
        m_cond2 = MConductor(cond2, conv, RED)

        field.compute()

        self.add(m_field.as_bg_image())
        self.add(m_field.as_quiver(4))
        self.add(m_cond1.as_rect())
        self.add(m_cond2.as_rect())

        gen1 = IonGenerator(PROTON_MASS, PROTON_CHARGE, (0, 0), (6.0, 10.0), (0.5, 1.0))
        m_gen1 = MIonGenerator(gen1, conv)

        gen2 = IonGenerator(PROTON_MASS, ELECTRON_CHARGE, (0, 0), (6.0, 10.0), (8.0, 8.5))
        m_gen2 = MIonGenerator(gen2, conv)

        sim = IonSimulation(field)
        m_sim = MIonSimulation(sim,conv)

        gr1 = [RED_E,RED_A]
        sim.simulate(gen1, 10, 500, 1e-5)
        paths1 = m_sim.as_paths(gr1)
        # objects1, animations1 = m_sim.as_at_once(gr1)

        gr2 = [BLUE_E,BLUE_A]
        sim.simulate(gen2, 10, 500, 1e-5)
        paths2 = m_sim.as_paths(gr2)
        # objects2, animations2 = m_sim.as_at_once(gr2)

        self.add(m_gen1.as_rect())
        self.add(m_gen2.as_rect())
        self.add(*paths1,*paths2)
        # self.add(*objects1,*objects2)
        # self.play(*animations1, *animations2 ,rate_func=linear, run_time=5)
        # self.wait()
