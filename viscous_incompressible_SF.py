import numpy as np
from .TorusDEC2D import TorusDEC2D

class viscous_incompressible_SF(TorusDEC2D):
    def __init__(self, sizex, sizey, resx, resy, nu, hbar, dt, eps_f=1e-8):
        super().__init__(sizex, sizey, resx, resy)
        self.nu = nu
        self.hbar = hbar
        self.dt = dt
        self.eps_f = eps_f          # regolarizzazione di Tikhonov per f
        self.schrodinger_mask = None
        self.viscous_mask = None

    def build_schrodinger(self):
        #maschera coerente con segno discreto
        nx, ny = self.resx, self.resy 

        lam = (
            -4*np.sin(np.pi * self.iix / nx)**2 / self.dx**2
            -4*np.sin(np.pi * self.iiy / ny)**2 / self.dy**2
        )
        
        self.schrodinger_mask = np.exp(1j * lam * self.hbar * self.dt / 2)
        self.viscous_mask = np.exp(lam * self.nu * self.dt)*self.schrodinger_mask

    #evoluzione temporale, con splitting temporale
    # 1. evoluzione cinetica con maschera a segno discreto
    # 2. correzione di potenziale locale (due approcci)
    def schrodinger_flow(self, psi1, psi2, correzione = "correzione esatta"):
        if self.schrodinger_mask is None:
            raise ValueError("Call build_schrodinger() before schrodinger_flow().")

        #passo temporale splittato
        psi1_hat = np.fft.fftn(psi1)
        psi2_hat = np.fft.fftn(psi2)

        psi1_hat = psi1_hat * self.viscous_mask
        psi2_hat = psi2_hat * self.viscous_mask

        psi1_star = np.fft.ifftn(psi1_hat)
        psi2_star = np.fft.ifftn(psi2_hat)

        #correzione del potenziale local... 
        # -> correzione on: metodo del paper, approssimato ()
        # -> correzione esatta: correzione alternativa, ma esatta
        if correzione == "correzione off": 
            return psi1_star.copy(), psi2_star.copy()

        #calcolo potenziali congelati
        vr = self.compute_vr(psi1, psi2)
        vi = self.compute_vi(psi1, psi2)
        #p_tilde = self.compute_p_tilde(psi1, psi2)
        p, q = self.compute_p_q(psi1, psi2)

        if correzione == "correzione on":
            phase1 = np.exp(-1j * (vr + p) * self.dt / self.hbar)
            phase2 = np.exp(-1j * (vr - p) * self.dt / self.hbar)

            coupling12 = np.exp(-1j * q * self.dt / self.hbar) - 1.0
            coupling21 = np.exp(-1j * np.conjugate(q) * self.dt / self.hbar) - 1.0

            psi1_new = np.exp(vi * self.dt / self.hbar)*(psi1_star * phase1 + psi2_star * coupling12)
            psi2_new = np.exp(vi * self.dt / self.hbar)*(psi2_star * phase2 + psi1_star * coupling21)

        elif correzione == "correzione esatta": 
            beta = np.sqrt(p**2 + np.abs(q)**2)
            phase = np.exp(vi * self.dt / self.hbar)*np.exp(-1j * vr * self.dt/self.hbar)

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
    
    def compute_p_q(self, psi1, psi2):
        sx, sy, sz = self.hopf(psi1, psi2)

        lap_sx = self.laplacian_from_dec(sx)
        lap_sy = self.laplacian_from_dec(sy)
        lap_sz = self.laplacian_from_dec(sz)

        f1, f2, f3 = self.compute_f(psi1, psi2)

        p = 0.25 * self.hbar**2 * lap_sz - self.hbar * f3
        q = 0.25 * self.hbar**2 * (lap_sx - 1j * lap_sy) - self.hbar * (f1 - 1j * f2)
        return p, q

    def compute_vr(self, psi1, psi2):
        vr = (self.grad_psi_norm_squared(psi1, psi2)*self.hbar**2) / 2
        return vr

    def compute_vi(self, psi1, psi2):
        # V_i = hbar * nu * |grad Psi|^2 / rho0   (NON hbar^2)
        return self.hbar * self.nu * self.grad_psi_norm_squared(psi1, psi2)
    
    def compute_f(self, psi1, psi2):
        """
        Risolve A f = S punto per punto, A (2x3) = [[d_x s1, d_x s2, d_x s3],
                                                    [d_y s1, d_y s2, d_y s3]].
        Pseudo-inversa regolarizzata (Tikhonov): f = A^T (A A^T + eps I)^{-1} S,
        con eps = eps_f * max(tr(A A^T)); A A^T e' 2x2 -> inversa analitica.
        """
        sx, sy, sz = self.hopf(psi1, psi2)
        ds1x, ds1y = self.gradient_from_dec(sx)
        ds2x, ds2y = self.gradient_from_dec(sy)
        ds3x, ds3y = self.gradient_from_dec(sz)
        Ax = np.stack([ds1x, ds2x, ds3x])             # riga x di A, shape (3, Nx, Ny)
        Ay = np.stack([ds1y, ds2y, ds3y])             # riga y di A

        ux, uy = self.velocity_nodes(psi1, psi2)
        g2 = self.grad_psi_norm_squared(psi1, psi2)

        gx = [self.gradient_from_dec(c) for c in (psi1, psi2)]          # d_k psi_c
        Mx = np.zeros(psi1.shape); My = np.zeros(psi1.shape)
        for (dx_, dy_) in gx:
            dxx, dxy = self.gradient_from_dec(dx_); dyx, dyy = self.gradient_from_dec(dy_)
            Mx += np.imag(np.conj(dx_)*dxx + np.conj(dy_)*dyx)
            My += np.imag(np.conj(dx_)*dxy + np.conj(dy_)*dyy)

        Sx = 2 * self.nu * g2 * ux / self.hbar -2*self.nu*Mx
        Sy = 2 * self.nu * g2 * uy / self.hbar -2*self.nu*My

        a = np.sum(Ax * Ax, axis=0)                   # A A^T = [[a, b], [b, d]]
        b = np.sum(Ax * Ay, axis=0)
        d = np.sum(Ay * Ay, axis=0)
        eps = self.eps_f * np.max(a + d)
        a, d = a + eps, d + eps
        det = a * d - b**2
        y1 = ( d * Sx - b * Sy) / det                 # (A A^T + eps I)^{-1} S
        y2 = (-b * Sx + a * Sy) / det
        f = Ax * y1 + Ay * y2                         # A^T y, shape (3, Nx, Ny)

        # diagnostica: residuo relativo di A f = S (≈0 dove A ha rango 2)
        rx = np.sum(Ax * f, axis=0) - Sx
        ry = np.sum(Ay * f, axis=0) - Sy
        self.last_f_residual = np.sqrt(np.mean(rx**2 + ry**2) /
                                       max(np.mean(Sx**2 + Sy**2), 1e-300))
        return f[0], f[1], f[2]
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
    
    def pressure_project(self, psi1, psi2):
        vx, vy = self.spinor_velocity_one_form(psi1, psi2)
        div = self.div_one_form(vx, vy)
        q = self.poisson_solve(div / self.hbar)
        psi1, psi2 = self.gauge_transform(psi1, psi2, -q)
        return psi1, psi2

    def vorticity_two_form(self, vx,vy):
        w = self.derivative_of_one_form(vx, vy)
        return w

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

    def velocity_nodes(self, psi1, psi2):
        # u ai nodi: la 1-forma vive sugli edge e vale u*dx -> sharp (divide per dx e media)
        return self.sharp_vortex(*self.spinor_velocity_one_form(psi1, psi2))
    
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

    def compute_pressure(self, psi1, psi2):
        # pressura classica (solo diagnostica): Lap p = -d_i u_j d_j u_i
        ux, uy = self.velocity_nodes(psi1, psi2)
        du_dx, du_dy = self.gradient_from_dec(ux)
        dv_dx, dv_dy = self.gradient_from_dec(uy)
        return self.poisson_solve(-(du_dx**2 + 2 * du_dy * dv_dx + dv_dy**2))