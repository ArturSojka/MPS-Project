from manim import *
from manim_physics import *
from simulation.simulation import ElectricField as EField, Conductor, IonGenerator, IonSimulation
import matplotlib.colors as colors
import matplotlib.cm as cm

class ElectricFieldTest(Scene):
    def construct(self):
        manim_coords = (config["frame_width"],config["frame_height"])
        field = EField((160,90),0.1,manim_coords)
        
        cond1 = Conductor(-1,(15,25),(40,50))
        field.add(cond1)
        c1_rect = cond1.rect()
        rect1 = Rectangle(width=c1_rect[0],height=c1_rect[1],color=BLUE)
        rect1.move_to(np.array([c1_rect[2],c1_rect[3],0.0]))
        
        cond2 = Conductor(1,(135,145),(40,50))
        field.add(cond2)
        c2_rect = cond2.rect()
        rect2 = Rectangle(width=c2_rect[0],height=c2_rect[1],color=RED)
        rect2.move_to(np.array([c2_rect[2],c2_rect[3],0.0]))
        
        field.compute()

        data = field.V[1:-1,1:-1].T
        norm = colors.Normalize(vmin=np.nanmin(data), vmax=np.nanmax(data))
        scalar_mappable = cm.ScalarMappable(norm=norm, cmap='RdBu_r')
        image = ImageMobject(scalar_mappable.to_rgba(data,bytes=True))
        image.set_resampling_algorithm(RESAMPLING_ALGORITHMS['cubic'])
        image.width = config["frame_width"]
        image.height = config["frame_height"]
        
        vfield = ArrowVectorField(
            field.get_field_at_manim, length_func=lambda x: x,
            x_range=[-8,8,0.2], y_range=[-4.5,4.5,0.2],
            color=BLACK, opacity=0.6
        )
        
        self.add(image)
        self.add(vfield)
        self.add(rect1)
        self.add(rect2)
        
        gen = IonGenerator(1.67e-27, 1.6e-19, np.array([0,30000]),(60,100),(5,10))
        n_ions = 10
        cols = color_gradient([GOLD_E,GOLD_A],n_ions)
        sim = IonSimulation(field,gen,n_ions)
        for _ in range(200):
            sim.step(1e-5)
        
        ions = []
        for i, c in enumerate(cols):
            # self.add(*[
            #     Dot(p,color=ORANGE)
            #     for p in sim.traj_as_manim_points(i)
            # ])
            path = VMobject(stroke_color=c)
            traj = sim.traj_as_manim_points(i)
            path.set_points_smoothly(traj)
            dot = Dot(traj[0],0.04,color=c)
            ions.append((path,dot))
            self.add(dot)
        self.play(*[
            MoveAlongPath(dot,path)
            for path, dot in ions
        ], rate_func=linear, duration=2)

class ElectricFieldExampleScene(Scene):
    def construct(self):
        charge1 = Charge(-1, LEFT)
        charge2 = Charge(-1, RIGHT)
        charge3 = Charge(2, UP)
        field = always_redraw( 
            lambda: ElectricField( 
                charge1, 
                charge2, 
                charge3 
            ) 
        ) 
        
        self.add(charge1, charge2, charge3)
        self.add(field)
        self.play(charge3.animate.move_to(DOWN))

class DefaultTemplate(Scene):
    def construct(self):
        circle = Circle()  # create a circle
        circle.set_fill(PINK, opacity=0.5)  # set color and transparency

        square = Square()  # create a square
        square.flip(RIGHT)  # flip horizontally
        square.rotate(-3 * TAU / 8)  # rotate a certain amount

        self.play(Create(square))  # animate the creation of the square
        self.play(Transform(square, circle))  # interpolate the square into the circle
        self.play(FadeOut(square))  # fade out animation

class ContinuousMotion(Scene):
    def construct(self):
        func = lambda pos: np.sin(pos[0] / 2) * UR + np.cos(pos[1] / 2) * LEFT
        stream_lines = StreamLines(func, stroke_width=3, max_anchors_per_line=30)
        self.add(stream_lines)
        stream_lines.start_animation(warm_up=False, flow_speed=1.5)
        self.wait(stream_lines.virtual_time / stream_lines.flow_speed)
