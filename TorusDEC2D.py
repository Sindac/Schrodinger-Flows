#provo a ricreare l'ambiete che ha fatto Chern, in cui esistevano operatori

import numpy as np


class TorusDEC2D:
    """
    Discrete Exterior Calculus on a 3D periodic grid (torus).

    This class stores the geometric data of the grid and provides
    differential operators acting on discrete scalar fields, 1-forms,
    and 2-forms.
    """

    def __init__(self, sizex, sizey, resx, resy):
        # Domain sizes
        self.sizex = float(sizex)
        self.sizey = float(sizey)

        # Grid resolutions
        self.resx = int(round(resx))
        self.resy = int(round(resy))

        # Grid spacing
        self.dx = self.sizex / self.resx
        self.dy = self.sizey / self.resy

        # 1D index arrays: MATLAB had 1:resx, here we use 0,...,resx-1
        self.ix = np.arange(self.resx)
        self.iy = np.arange(self.resy)

        # 3D index arrays, equivalent to MATLAB ndgrid
        self.iix, self.iiy, = np.meshgrid(
            self.ix, self.iy, indexing="ij"
        )

        # Grid point coordinates
        self.px = self.iix * self.dx
        self.py = self.iiy * self.dy

    def derivative_of_function(self, f):
        
        #For a scalar function f, compute the discrete 1-form df.
        #Periodicity is enforced with np.roll.
        
        vx = np.roll(f, shift=-1, axis=0) - f
        vy = np.roll(f, shift=-1, axis=1) - f
        return vx, vy

    def derivative_of_one_form(self, vx, vy):
        
        #For a discrete 1-form (vx, vy, vz), compute the discrete 2-form dv.

        w = vx - np.roll(vx, shift=-1, axis=1) + np.roll(vy, shift=-1, axis=0) - vy
        return w

        #PENSO CHE LA DERIVATA DI UNA 2-forma IN 2D SIA 0
    """def derivative_of_two_form(self, wx, wy):
        
        #For a discrete 2-form (wx, wy, wz), compute the discrete 3-form dw.
        
        f = np.roll(wx, shift=-1, axis=0) - wx
        f = f + np.roll(wy, shift=-1, axis=1) - wy
        return f"""

    """in 2D ha ancora senso l'operatore hodge: 0-forma <--> 2-forma, 1-forma <--> 1-forma
        Bisogna però vedere la divergenza che forma assume"""

    def div_one_form(self, vx, vy):        
        f = (vx - np.roll(vx, shift=1, axis=0)) / (self.dx ** 2)
        f = f + (vy - np.roll(vy, shift=1, axis=1)) / (self.dy ** 2)
        return f

    def sharp_vortex(self, vx, vy):
        
        #For a discrete 1-form v, compute the corresponding vector field v^sharp
        #by averaging to vertices.
        
        ux = 0.5 * (np.roll(vx, shift=1, axis=0) + vx) / self.dx
        uy = 0.5 * (np.roll(vy, shift=1, axis=1) + vy) / self.dy
        return ux, uy

    def sharp_staggered(self, vx, vy):
        
        #For a discrete 1-form v, compute the corresponding staggered vector field.
        
        ux = vx / self.dx
        uy = vy / self.dy
        return ux, uy

    def poisson_solve(self, f):
        
        #Solve the Poisson equation by spectral method on the periodic grid.
    
        f_hat = np.fft.fftn(f)

        sx = np.sin(np.pi * self.iix / self.resx) / self.dx
        sy = np.sin(np.pi * self.iiy / self.resy) / self.dy

        denom = sx**2 + sy**2
        fac = np.zeros_like(denom, dtype=np.complex128)

        mask = denom != 0
        fac[mask] = -0.25 / denom[mask]
        fac[0, 0] = 0.0

        u_hat = f_hat * fac
        u = np.fft.ifftn(u_hat)

        # If the input is real, the imaginary part should only be numerical noise
        return np.real(u)
