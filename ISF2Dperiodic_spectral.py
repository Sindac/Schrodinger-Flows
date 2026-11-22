import numpy as np
from .TorusDEC2D import TorusDEC2D

class ISF2Dperiodic_spectral(TorusDEC2D):
    def __init__(self, sizex, sizey, resx, resy, hbar, dt):
        super().__init__(sizex, sizey, resx, resy)
        self.hbar = hbar
        self.dt = dt
        self.schrodinger_mask = None

    def build_schrodinger(self):
        nx, ny = self.resx, self.resy
        fac = -4 * np.pi**2 * self.hbar

        # Traduzione diretta dell'idea MATLAB
        kx = (self.ix - nx / 2) / self.sizex
        ky = (self.iy - ny / 2) / self.sizey
        self.KX, self.KY = np.meshgrid(kx, ky, indexing="ij")

        lam = fac * (self.KX**2 + self.KY**2)
        self.schrodinger_mask = np.exp(1j * lam * self.dt / 2)

    def schrodinger_flow(self, psi1, psi2, correzione = "correzione esatta"):
        if self.schrodinger_mask is None:
            raise ValueError("Call build_schrodinger() before schrodinger_flow().")

        psi1_hat = np.fft.fftshift(np.fft.fftn(psi1))
        psi2_hat = np.fft.fftshift(np.fft.fftn(psi2))

        psi1_hat = psi1_hat * self.schrodinger_mask
        psi2_hat = psi2_hat * self.schrodinger_mask

        psi1_star = np.fft.ifftn(np.fft.ifftshift(psi1_hat))
        psi2_star = np.fft.ifftn(np.fft.ifftshift(psi2_hat))

        #fino a qua ok, perchè trotter-suzuki permette di risolvere la parte di laplaciano così
        #ora devo aggiungere P, Vr, Q

        #NOTA: adesso ok, ma forse sarà meglio calcolarlo da un'altra parte e poi usarlo come valori trovati 
        #all'inizio e buttati qua dentro perchè \psi usati per s sono quelli dell'inizio
        

        if correzione == "correzione off": 
            psi1_new = psi1_star.copy()
            psi2_new = psi2_star.copy()

        elif correzione == "correzione on":
            vr_eff = self.compute_vr_eff(psi1, psi2)
            
            p, q = self.compute_p_q(psi1, psi2)

            phase1 = np.exp(-1j * (vr_eff + p) * self.dt)
            phase2 = np.exp(-1j * (vr_eff - p) * self.dt)

            coupling12 = np.exp(-1j * q * self.dt) - 1.0
            coupling21 = np.exp(-1j * np.conjugate(q) * self.dt) - 1.0

            psi1_new = psi1_star * phase1 + psi2_star * coupling12
            psi2_new = psi2_star * phase2 + psi1_star * coupling21

        elif correzione == "correzione esatta":
            vr = self.compute_vr_eff(psi1, psi2)
            
            p, q = self.compute_p_q(psi1, psi2)

            beta = np.sqrt(p**2 + np.abs(q)**2)
            phase = np.exp(-1j * vr * self.dt)

            c = np.cos(beta * self.dt)

            s_over_beta = np.zeros_like(beta, dtype=np.complex128)
            mask = beta > 1e-14
            s_over_beta[mask] = np.sin(beta[mask] * self.dt) / beta[mask]
            s_over_beta[~mask] = self.dt

            psi1_new = phase * (
            c * psi1_star
            - 1j * s_over_beta * (p * psi1_star + q * psi2_star)
            )

            psi2_new = phase * (
                c * psi2_star
                - 1j * s_over_beta * (np.conjugate(q) * psi1_star - p * psi2_star)
            )
        else:
            raise ValueError("correzione must be 'correzione on', 'correzione off' o 'correzione esatta'.")
             

        return psi1_new, psi2_new
    
    def compute_p_q(self, psi1, psi2):
        sx, sy, sz = self.hopf(psi1, psi2)

        lap_sx = self.laplacian(sx) #CAMBIA LAPLATIAN
        lap_sy = self.laplacian(sy)
        lap_sz = self.laplacian(sz)

        p = 0.25 * self.hbar**2 * lap_sz
        q = 0.25 * self.hbar**2 * (lap_sx - 1j * lap_sy)

        return p, q

    def compute_vr_eff(self, psi1, psi2):
        vr = (self.grad_psi_norm_squared(psi1, psi2)*self.hbar**2) / 2     

        """ux, uy = self.spinor_velocity_one_form(psi1, psi2)
        ux, uy = self.sharp_staggered(ux,uy)

        duxdx, duxdy = self.gradient(ux)
        duydx, duydy = self.gradient(uy)

        rhs_p = (
            duxdx**2
            + 2.0 * (duxdy * duydx)
            + duydy**2
        )
        rhs_p = rhs_p - np.mean(rhs_p)
        p_phys = self.poisson_solve(-rhs_p)
        p_phys -= np.mean(p_phys)

        vr_eff = vr + p_phys - 0.5 * (ux**2 + uy**2)"""

        return vr
    
    def spinor_velocity_one_form(self, psi1, psi2):
        vx = np.angle(
            np.conjugate(psi1) * np.roll(psi1, shift=-1, axis=0) +
            np.conjugate(psi2) * np.roll(psi2, shift=-1, axis=0)
        )
        vy = np.angle(
            np.conjugate(psi1) * np.roll(psi1, shift=-1, axis=1) +
            np.conjugate(psi2) * np.roll(psi2, shift=-1, axis=1)
        )

        vx = vx * self.hbar
        vy = vy * self.hbar
        return vx, vy
    
    def vorticity_two_form(self, vx,vy):
        w = self.derivative_of_one_form(vx, vy)
        return w

    def gradient(self, f):
        f_hat = np.fft.fftshift(np.fft.fftn(f))
        dfdx_hat = (2j * np.pi * self.KX) * f_hat
        dfdy_hat = (2j * np.pi * self.KY) * f_hat
        dfdx = np.fft.ifftn(np.fft.ifftshift(dfdx_hat))
        dfdy = np.fft.ifftn(np.fft.ifftshift(dfdy_hat))
        return dfdx, dfdy

    def laplacian(self, f):
        f_hat = np.fft.fftshift(np.fft.fftn(f))
        lap_hat = (-4 * np.pi**2) * (self.KX**2 + self.KY**2) * f_hat
        lap = np.fft.ifftn(np.fft.ifftshift(lap_hat))
        return lap

    
    """
    def gradient(self, f):
        dfdx = (np.roll(f, shift=-1, axis=0) - np.roll(f, shift=1, axis=0)) / (2 * self.dx)
        dfdy = (np.roll(f, shift=-1, axis=1) - np.roll(f, shift=1, axis=1)) / (2 * self.dy)
        return dfdx, dfdy
    
    
    def laplacian(self, f):
        d2fdx2 = (
            np.roll(f, shift=-1, axis=0)
            - 2 * f
            + np.roll(f, shift=1, axis=0)
        ) / (self.dx ** 2)

        d2fdy2 = (
            np.roll(f, shift=-1, axis=1)
            - 2 * f
            + np.roll(f, shift=1, axis=1)
        ) / (self.dy ** 2)

        return d2fdx2 + d2fdy2
    """
    
    def grad_psi_norm_squared(self, psi1, psi2):
        dpsi1_dx, dpsi1_dy = self.gradient(psi1)
        dpsi2_dx, dpsi2_dy = self.gradient(psi2)

        norm_grad_psi1 = (
            np.abs(dpsi1_dx) ** 2
            + np.abs(dpsi1_dy) ** 2
        )

        norm_grad_psi2 = (
            np.abs(dpsi2_dx) ** 2
            + np.abs(dpsi2_dy) ** 2
        )

        return norm_grad_psi1 + norm_grad_psi2
    
    def pressure_project(self, psi1, psi2):
        vx, vy = self.spinor_velocity_one_form(psi1, psi2)
        div = self.div_one_form(vx, vy)
        q = self.poisson_solve(div / self.hbar)
        psi1, psi2 = self.gauge_transform(psi1, psi2, -q)
        return psi1, psi2
    
    def advance(self, psi1, psi2, correzione="correzione esatta"):
        psi1, psi2 = self.schrodinger_flow(psi1, psi2, correzione=correzione)
        psi1, psi2 = self.normalize(psi1, psi2)
        psi1, psi2 = self.pressure_project(psi1, psi2)
        return psi1, psi2
    
    @staticmethod
    def gauge_transform(psi1, psi2, q):
        eiq = np.exp(1j * q)
        psi1 = psi1 * eiq
        psi2 = psi2 * eiq
        return psi1, psi2

    @staticmethod
    def hopf(psi1, psi2):
        a = np.real(psi1)
        b = np.imag(psi1)
        c = np.real(psi2)
        d = np.imag(psi2)

        sx = 2 * (a * c + b * d)
        sy = 2 * (a * d - b * c)
        sz = a**2 + b**2 - c**2 - d**2
        return sx, sy, sz
    
    @staticmethod
    def normalize(psi1, psi2):
        psi_norm = np.sqrt(np.abs(psi1)**2 + np.abs(psi2)**2)
        psi1 = psi1 / psi_norm
        psi2 = psi2 / psi_norm
        return psi1, psi2