import numpy as np
from typing import List, Tuple

class Conductor:
    
    def __init__(self, voltage: float, x_range: Tuple[int,int], y_range: Tuple[int,int]):
        self.x1, self.x2 = x_range
        self.y1, self.y2 = y_range
        self.voltage = voltage
        
    def rect(self):
        p1 = self.p2c(self.x1,self.y1)
        p2 = self.p2c(self.x2,self.y2)
        return (
            p2[0]-p1[0],p2[1]-p1[1],
            (p1[0]+p2[0])/2,(p1[1]+p2[1])/2
        )

class ElectricField:
    
    def __init__(self, grid_size: Tuple[int,int], dx: float, manim_coords: Tuple[float,float]):
        self.nx, self.ny = grid_size
        self.dx = dx
        self.manim_coords = manim_coords

        self.V = np.zeros((self.nx+2, self.ny+2))
        self.conductor_mask = np.zeros((self.nx, self.ny), dtype=bool)
        
        self.Ex = np.zeros((self.nx, self.ny))
        self.Ey = np.zeros((self.nx, self.ny))
    
    def add(self, c: Conductor):
        self.conductor_mask[c.x1:c.x2,c.y1:c.y2] = True
        self.V[c.x1+1:c.x2+1,c.y1+1:c.y2+1] = c.voltage
        c.p2c = self.p2c
        
    def compute(self, it=100000, eps=1e-5):
        for i in range(it):
            V_old = self.V.copy()
            
            self.V[1:-1,1:-1] = np.where( # Dirichlet boundry condition
                self.conductor_mask,
                self.V[1:-1,1:-1],
                (self.V[2:,1:-1]+self.V[:-2,1:-1]+self.V[1:-1,2:]+self.V[1:-1,:-2])*0.25
            )
            
            delta = np.max(np.abs(self.V - V_old))
            if delta < eps:
                print(f"Converged in {i} iterations")
                break
        
        self.Ex = np.where(
            self.conductor_mask,
            0,
            -(self.V[2:, 1:-1] - self.V[:-2, 1:-1]) / (2 * self.dx)
        )
        self.Ey = np.where(
            self.conductor_mask,
            0,
            -(self.V[1:-1, 2:] - self.V[1:-1, :-2]) / (2 * self.dx)
        )

    def get_field_at(self, x: float, y: float) -> Tuple[float, float]:
        i = x / self.dx
        j = y / self.dx
        
        if i < 0 or i >= self.nx - 1 or j < 0 or j >= self.ny - 1:
            return (0,0)
        
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
        
        return (w_x @ Ex_mat @ w_y, w_x @ Ey_mat @ w_y)
    
    def c2p(self, x: float, y: float) -> np.ndarray:
        return np.array([
            (x+self.manim_coords[0] / 2)*(self.nx*self.dx/self.manim_coords[0]),
            (y+self.manim_coords[1] / 2)*(self.ny*self.dx/self.manim_coords[1]),
            0.0
        ])
        
    def p2c(self,x: float, y: float) -> np.ndarray:
        return np.array([
            (x-self.nx/2)*(self.manim_coords[0]/self.nx),
            (y-self.ny/2)*(self.manim_coords[1]/self.ny),
            0.0
        ])
    
    def get_field_at_manim(self, point: np.ndarray) -> np.ndarray:
        p = self.c2p(point[0],point[1])
        return np.array([*self.get_field_at(p[0],p[1]),0.0])

    def get_field_at_batch(self, x_arr: np.ndarray, y_arr:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        i_arr = np.clip(x_arr / self.dx,0,self.nx-1)
        j_arr = np.clip(y_arr / self.dx,0,self.ny-1)
        
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
    
    def __init__(self, mass: float, charge: float, init_v: Tuple[float,float], x_range: Tuple[float,float], y_range: Tuple[float,float]):
        self.x1, self.x2 = x_range
        self.y1, self.y2 = y_range
        self.mass = mass
        self.charge = charge
        self.init_v = init_v
        
    def generate_initial(self, n: int, dx: float):
        xs = (np.random.random((n,))*(self.x2-self.x1)+self.x1)*dx
        ys = (np.random.random((n,))*(self.y2-self.y1)+self.y1)*dx
        
        vxs = (np.random.random((n,))*0.05+self.init_v[0])*dx
        vys = (np.random.random((n,))*0.05+self.init_v[1])*dx
        
        return (xs,ys,vxs,vys)
        
class IonSimulation:
    
    def __init__(self,field: ElectricField, generator: IonGenerator, n: int):
        self.field = field
        self.generator = generator
        
        self.x_pos, self.y_pos, self.x_vel, self.y_vel = generator.generate_initial(n,field.dx)
        self.trajectories = [(self.x_pos,self.y_pos)]
        
    def step(self, dt: float):
        Ex, Ey = self.field.get_field_at_batch(self.x_pos, self.y_pos)
        ax = self.generator.charge * Ex / self.generator.mass
        ay = self.generator.charge * Ey / self.generator.mass
        
        self.x_vel = self.x_vel + ax * dt
        self.y_vel = self.y_vel + ay * dt
        
        self.x_pos = self.x_pos + self.x_vel * dt
        self.y_pos = self.y_pos + self.y_vel * dt
        
        self.trajectories.append((self.x_pos,self.y_pos))
        
    def traj_as_manim_points(self, i: int) -> List[np.ndarray]:
        return [
            self.field.p2c(x[i]/self.field.dx,y[i]/self.field.dx)
            for x,y in self.trajectories
        ]

