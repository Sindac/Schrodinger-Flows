import numpy as np
from .TorusDEC2D import TorusDEC2D

class ISF2D_DECbased(TorusDEC2D):
    def __init__(self, sizex, sizey, resx, resy, hbar, dt):
        super().__init__(sizex, sizey, resx, resy)
        self.hbar = hbar
        self.dt = dt
        self.schrodinger_mask = None

    def build_schrodinger(self):
        #maschera coerente con segno discreto
        nx, ny = self.resx, self.resy 

        lam = (
            -4*np.sin(np.pi * self.iix / nx)**2 / self.dx**2
            -4*np.sin(np.pi * self.iiy / ny)**2 / self.dy**2
        )
        
        self.schrodinger_mask = np.exp(1j * lam * self.hbar * self.dt / 2)

    #evoluzione temporale, con splitting temporale
    # 1. evoluzione cinetica con maschera a segno discreto
    # 2. correzione di potenziale locale (due approcci)
    def schrodinger_flow(self, psi1, psi2, correzione = "correzione esatta"):
        if self.schrodinger_mask is None:
            raise ValueError("Call build_schrodinger() before schrodinger_flow().")

        #passo temporale splittato
        psi1_hat = np.fft.fftn(psi1)
        psi2_hat = np.fft.fftn(psi2)

        psi1_hat = psi1_hat * self.schrodinger_mask
        psi2_hat = psi2_hat * self.schrodinger_mask

        psi1_star = np.fft.ifftn(psi1_hat)
        psi2_star = np.fft.ifftn(psi2_hat)

        #correzione del potenziale local... 
        # -> correzione on: metodo del paper, approssimato ()
        # -> correzione esatta: correzione alternativa, ma esatta
        if correzione == "correzione off": 
            psi1_new = psi1_star.copy()
            psi2_new = psi2_star.copy()

        elif correzione == "correzione on":
            vr = self.compute_vr(psi1, psi2)
            
            p, q = self.compute_p_q(psi1, psi2)

            phase1 = np.exp(-1j * (vr + p) * self.dt)
            phase2 = np.exp(-1j * (vr - p) * self.dt)

            coupling12 = np.exp(-1j * q * self.dt) - 1.0
            coupling21 = np.exp(-1j * np.conjugate(q) * self.dt) - 1.0

            psi1_new = psi1_star * phase1 + psi2_star * coupling12
            psi2_new = psi2_star * phase2 + psi1_star * coupling21

        elif correzione == "correzione esatta":
            vr = self.compute_vr(psi1, psi2)
            
            p, q = self.compute_p_q(psi1, psi2)

            beta = np.sqrt(p**2 + np.abs(q)**2)
            phase = np.exp(-1j * vr * self.dt/self.hbar)

            argument = beta * self.dt/self.hbar
            c = np.cos(argument)

            s_over_beta = np.zeros_like(beta, dtype=np.complex128)
            mask = beta > 1e-14
            s_over_beta[mask] = (np.sin(argument[mask]) / beta[mask])
            s_over_beta[~mask] = self.dt/self.hbar  # Handle the case when beta is very small

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
    
    #computa i termini correttivi del potenziale, che a livello teorico         | [P   Q]
    #sono gli elementi della matrice(2x2)/termine stern gerlac  --------------> | [Q* -P]
    def compute_p_q(self, psi1, psi2):
        sx, sy, sz = self.hopf(psi1, psi2)

        lap_sx = self.laplacian_from_dec(sx)
        lap_sy = self.laplacian_from_dec(sy)
        lap_sz = self.laplacian_from_dec(sz)

        p = 0.25 * self.hbar**2 * lap_sz
        q = 0.25 * self.hbar**2 * (lap_sx - 1j * lap_sy)

        return p, q
    
    #altro termine correttivo del potenziale
    def compute_vr(self, psi1, psi2):
        vr = (self.grad_psi_norm_squared(psi1, psi2)*self.hbar**2) / 2
        return vr
    
    #vincolo di incomprimibilità 
    def pressure_project(self, psi1, psi2):
        vx, vy = self.spinor_velocity_one_form(psi1, psi2)
        div = self.div_one_form(vx, vy)
        q = self.poisson_solve(div / self.hbar)
        psi1, psi2 = self.gauge_transform(psi1, psi2, -q)
        return psi1, psi2
    
    #link geometrico che vive sugli edge
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
    
    #quadrato della norma del gradiente di psi
    def grad_psi_norm_squared(self, psi1, psi2):
        dpsi1_dx, dpsi1_dy = self.gradient_from_dec(psi1)
        dpsi2_dx, dpsi2_dy = self.gradient_from_dec(psi2)

        norm_grad_psi1 = (
            np.abs(dpsi1_dx) ** 2
            + np.abs(dpsi1_dy) ** 2
        )

        norm_grad_psi2 = (
            np.abs(dpsi2_dx) ** 2
            + np.abs(dpsi2_dy) ** 2
        )

        return norm_grad_psi1 + norm_grad_psi2
    
    #gadiente 
    def gradient_from_dec(self, f):
        #d(psi) 1-forma vive sugli edge. non posso elevarli al quadrato e sommarli perchè non sarebbe locale
        # quindi sharp_vortex media ai nodi; ora posso elevare e sommare perchè sono puntuli
        vx, vy = self.derivative_of_function(f)
        return self.sharp_vortex(vx, vy)
    
    def laplacian_from_dec(self, f):
        #laplacian = div_h(d_h f) -> scalare ... 0-forma!
        vx, vy = self.derivative_of_function(f)
        return self.div_one_form(vx,vy)
    
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

    #calcolo delle componenti del vettore spin
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