import numpy as np
from typing import List, Tuple, overload

class CoordsConverter:
    
    def __init__(self, array_range: Tuple[int,int], real_range: Tuple[float,float], dx: float, manim_range: Tuple[float,float]):
        self.array_range = array_range
        self.real_range = real_range
        self.dx = dx
        self.manim_range = manim_range
        
    @overload
    def real2array(self, x: float, y: float) -> Tuple[int,int]:
        ...
        
    @overload
    def real2array(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
        ...
        
    def real2array(self, x: float | np.ndarray, y: float | np.ndarray) -> Tuple[int,int] | Tuple[np.ndarray,np.ndarray]:
        if isinstance(x, (float, int)):
            i = min(max(0,int(x / self.dx)),self.array_range[0]-1)
            j = min(max(0,int(y / self.dx)),self.array_range[1]-1)
            
            return i,j

        i_arr = np.clip(x / self.dx,0,self.array_range[0]-1).astype(int)
        j_arr = np.clip(y / self.dx,0,self.array_range[1]-1).astype(int)
        
        return i_arr, j_arr
    
    @overload
    def array2real(self, x: int, y: int) -> Tuple[float,float]:
        ...
        
    @overload
    def array2real(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
        ...
    
    def array2real(self, x: int | np.ndarray, y: int | np.ndarray) -> Tuple[float,float] | Tuple[np.ndarray,np.ndarray]:
        return x * self.dx, y * self.dx
    
    @overload
    def manim2real(self, x_or_points: float, y: float) -> Tuple[float,float]:
        ...
    
    @overload
    def manim2real(self, x_or_points: List[np.ndarray], y: None = None) -> List[Tuple[float,float]]:
        ...
    
    def manim2real(self, x_or_points: float | List[np.ndarray], y: float | None = None) -> Tuple[float,float] | List[Tuple[float,float]]:
        if isinstance(x_or_points, (float, int)):
            return (
                (x_or_points+self.manim_coords[0] / 2)*(self.array_range[0]*self.dx/self.manim_coords[0]),
                (y          +self.manim_coords[1] / 2)*(self.array_range[1]*self.dx/self.manim_coords[1])
            )
            
        return [(
            (p[0]+self.manim_coords[0] / 2)*(self.array_range[0]*self.dx/self.manim_coords[0]),
            (p[1]+self.manim_coords[1] / 2)*(self.array_range[1]*self.dx/self.manim_coords[1])
        ) for p in x_or_points]
        
    @overload
    def real2manim(self, x: float, y: float) -> np.ndarray:
        ...
    
    @overload
    def real2manim(self, x: np.ndarray, y: np.ndarray) -> List[np.ndarray]:
        ...
    
    def real2manim(self, x: float | np.ndarray, y: float | np.ndarray) -> np.ndarray | List[np.ndarray]:
        if isinstance(x, (float, int)):
            return np.array([
                (x-self.array_range[0]/2)*(self.manim_coords[0]/(self.array_range[0]*self.dx)),
                (y-self.array_range[1]/2)*(self.manim_coords[1]/(self.array_range[1]*self.dx)),
                0.0
            ])
            
        return [np.array([
            (xp-self.array_range[0]/2)*(self.manim_coords[0]/(self.array_range[0]*self.dx)),
            (yp-self.array_range[1]/2)*(self.manim_coords[1]/(self.array_range[1]*self.dx)),
            0.0
        ]) for xp, yp in zip(x,y)]
        
    @overload
    def manim2array(self, x_or_points: float, y: float) -> Tuple[int,int]:
        ...
    
    @overload
    def manim2array(self, x_or_points: List[np.ndarray], y: None = None) -> Tuple[np.ndarray,np.ndarray]:
        ...
    
    def manim2array(self, x_or_points: float | List[np.ndarray], y: float | None = None) -> Tuple[int,int] | Tuple[np.ndarray,np.ndarray]:
        if isinstance(x_or_points, (float, int)):
            r = self.manim2real(x_or_points,y)
            return self.real2array(*r)
        
        r = np.array(self.manim2real(x_or_points,y))
        return self.real2array(r[:,0],r[:,1])

    @overload
    def array2manim(self, x: int, y: int) -> np.ndarray:
        ...
    
    @overload
    def array2manim(self, x: np.ndarray, y: np.ndarray) -> List[np.ndarray]:
        ...
    
    def array2manim(self, x: int | np.ndarray, y: int | np.ndarray) -> np.ndarray | List[np.ndarray]:
        r = self.array2real(x,y)
        return self.real2manim(*r)
        