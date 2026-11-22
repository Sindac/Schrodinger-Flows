import numpy as np
from .TorusDEC2D import TorusDEC2D

class viscous_compressible_SF(TorusDEC2D):
    def __init__(self, sizex, sizey, resx, resy, mu, Mach, hbar, dt, U_ref=1.0, eps_f=1e-8,
                 vi_consistente=True):
        super().__init__(sizex, sizey, resx, resy)
        self.mu = mu
        self.Mach = Mach 
        self.c = U_ref / Mach   
        self.hbar = hbar
        self.dt = dt
        self.U_ref = U_ref
        self.eps_f = eps_f
        self.vi_consistente = vi_consistente   # vedi compute_vi
        self.schrodinger_mask = None
        self.viscous_mask = None
        self.lam = None
        self.last_f_residual = np.nan
        self.last_Pq_solenoidal = np.nan
        self.last_muq_solenoidal = np.nan

    def build_schrodinger(self):
        #maschera coerente con segno discreto
        nx, ny = self.resx, self.resy 
        self.lam = (
            -4*np.sin(np.pi * self.iix / nx)**2 / self.dx**2
            -4*np.sin(np.pi * self.iiy / ny)**2 / self.dy**2
        )
        self.schrodinger_mask = np.exp(1j * self.lam * self.hbar * self.dt / 2)
        # la parte viscosa (mu/rho) Lap dipende da rho(x): non e' diagonale in Fourier.
        # In schrodinger_flow: mu/rho_media in Fourier (esatto) + resto mu(1/rho - 1/rho_media) esplicito.

    #evoluzione temporale, con splitting temporale
    # 1. evoluzione cinetica con maschera a segno discreto
    # 2. correzione di potenziale locale (due approcci)
    def schrodinger_flow(self, psi1, psi2, correzione = "correzione esatta"):
        if self.schrodinger_mask is None:
            raise ValueError("Call build_schrodinger() before schrodinger_flow().")

        #passo temporale splittato
        # d psi/dt = (i hbar/2 + mu/rho) Lap psi
        rho = self.compute_rho(psi1, psi2)
        rho_media = np.mean(rho)
        self.viscous_mask = np.exp(self.lam * self.mu / rho_media * self.dt) * self.schrodinger_mask

        psi1_star = np.fft.ifftn(np.fft.fftn(psi1) * self.viscous_mask)
        psi2_star = np.fft.ifftn(np.fft.fftn(psi2) * self.viscous_mask)

        # resto a coefficiente variabile, esplicito (piccolo: rho - rho_media = O(Mach^2))
        resto = self.dt * self.mu * (1.0 / rho - 1.0 / rho_media)
        psi1_star = psi1_star + resto * self.laplacian_from_dec(psi1)
        psi2_star = psi2_star + resto * self.laplacian_from_dec(psi2)

        #correzione del potenziale local... 
        # -> correzione on: metodo del paper, approssimato ()
        # -> correzione esatta: correzione alternativa, ma esatta
        if correzione == "correzione off": 
            return psi1_star.copy(), psi2_star.copy()

        #calcolo potenziali congelati
        f = self.compute_f(psi1, psi2)          # una sola volta per passo
        vr = self.compute_vr(psi1, psi2)
        vi = self.compute_vi(psi1, psi2)
        p_tilde = self.compute_p_tilde(psi1, psi2, f)
        p, q = self.compute_p_q(psi1, psi2, f)

        if correzione == "correzione on":
            phase1 = np.exp(-1j * (vr + p) * self.dt / self.hbar)
            phase2 = np.exp(-1j * (vr - p) * self.dt / self.hbar)

            coupling12 = np.exp(-1j * q * self.dt / self.hbar) - 1.0
            coupling21 = np.exp(-1j * np.conjugate(q) * self.dt / self.hbar) - 1.0

            psi1_new = np.exp((vi-1j*p_tilde) * self.dt / self.hbar)*(psi1_star * phase1 + psi2_star * coupling12)
            psi2_new = np.exp((vi-1j*p_tilde) * self.dt / self.hbar)*(psi2_star * phase2 + psi1_star * coupling21)

        elif correzione == "correzione esatta":
            beta = np.sqrt(p**2 + np.abs(q)**2)
            phase = np.exp((vi-1j*p_tilde) * self.dt / self.hbar)*np.exp(-1j * vr * self.dt/self.hbar)

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

    def compute_rho(self, psi1, psi2):
        rho = np.abs(psi1)**2 + np.abs(psi2)**2
        return rho

    def compute_vr(self, psi1, psi2):
        # hbar^2 |grad Psi|^2 / (2 rho)
        return self.hbar**2 * self.grad_psi_norm_squared(psi1, psi2) / (2 * self.compute_rho(psi1, psi2))

    def compute_p_q(self, psi1, psi2, f=None):
        sx, sy, sz = self.hopf(psi1, psi2)

        lap_sx = self.laplacian_from_dec(sx)
        lap_sy = self.laplacian_from_dec(sy)
        lap_sz = self.laplacian_from_dec(sz)

        f1, f2, f3 = self.compute_f(psi1, psi2) if f is None else f

        p = 0.25 * self.hbar**2 * lap_sz / self.compute_rho(psi1, psi2) - self.hbar * f3
        q = 0.25 * self.hbar**2 * (lap_sx - 1j * lap_sy) / self.compute_rho(psi1, psi2) - self.hbar * (f1 - 1j * f2)
        return p, q

    def compute_vi(self, psi1, psi2):
        # V_i = (hbar mu / 2) (2 |grad Psi|^2 - Lap rho) / rho^2   (forma del paper)
        # Compensa punto per punto la perdita di densita' della diffusione immaginaria (mu/rho) Lap.
        # Nel continuo 2|grad Psi|^2 - Lap rho = -2 Re(Psi^dag Lap Psi): la forma "consistente"
        # usa lo STESSO laplaciano discreto del passo cinetico, cosi' la compensazione e' esatta
        # anche a livello discreto (la massa deriva solo per lo splitting, O(dt)).
        rho = self.compute_rho(psi1, psi2)
        if self.vi_consistente:
            re = np.real(np.conjugate(psi1) * self.laplacian_from_dec(psi1)
                         + np.conjugate(psi2) * self.laplacian_from_dec(psi2))
            return -self.hbar * self.mu * re / rho**2
        lap_rho = self.laplacian_from_dec(rho)
        return 0.5 * self.hbar * self.mu * (2 * self.grad_psi_norm_squared(psi1, psi2) - lap_rho) / rho**2

    def compute_p_tilde(self, psi1, psi2, f=None):
        vx, vy = self.velocity_nodes(psi1, psi2)
        kin = 0.5 * (np.abs(vx)**2 + np.abs(vy)**2)

        sx, sy, sz = self.hopf(psi1, psi2)
        f1, f2, f3 = self.compute_f(psi1, psi2) if f is None else f
        text = self.hbar / self.compute_rho(psi1, psi2) * (sx*f1 + sy*f2 + sz*f3)

        enthalpy = self.compute_enthalpy(psi1, psi2)

        P_quantum = self.compute_P_quantum(psi1, psi2) 

        mu_int = self.compute_mu_quantum(psi1, psi2)
        return enthalpy - P_quantum - kin + text - mu_int


    def compute_f(self, psi1, psi2):
        """
        Risolve A f = S punto per punto, A (2x3) = [[d_x s1, d_x s2, d_x s3],
                                                    [d_y s1, d_y s2, d_y s3]].
        Pseudo-inversa regolarizzata (Tikhonov): f = A^T (A A^T + eps I)^{-1} S,
        con eps = eps_f * max(tr(A A^T)); A A^T e' 2x2 -> inversa analitica.
        """
        # div u dalla 1-forma (valori sugli edge): div_one_form vuole la 1-forma, NON i nodi
        divu = self.div_one_form(*self.spinor_velocity_one_form(psi1, psi2))

        vx, vy = self.velocity_nodes(psi1, psi2)
        d_vx_x, d_vx_y = self.gradient_from_dec(vx)
        d_vy_x, d_vy_y = self.gradient_from_dec(vy)

        sx, sy, sz = self.hopf(psi1, psi2)
        ds1x, ds1y = self.gradient_from_dec(sx)
        ds2x, ds2y = self.gradient_from_dec(sy)
        ds3x, ds3y = self.gradient_from_dec(sz)

        ds1xx, ds1xy = self.gradient_from_dec(ds1x)
        ds1yx, ds1yy = self.gradient_from_dec(ds1y)
        ds2xx, ds2xy = self.gradient_from_dec(ds2x)
        ds2yx, ds2yy = self.gradient_from_dec(ds2y)
        ds3xx, ds3xy = self.gradient_from_dec(ds3x)
        ds3yx, ds3yy = self.gradient_from_dec(ds3y)

        drx, dry = self.gradient_from_dec(self.compute_rho(psi1, psi2))
        Drho = np.stack([drx, dry])  # shape (2, Nx, Ny)

        d_rdrx_x, d_rdrx_y = self.gradient_from_dec(self.compute_rho(psi1, psi2)*drx)
        d_rdry_x, d_rdry_y = self.gradient_from_dec(self.compute_rho(psi1, psi2)*dry)

        d_ln_rho_x, d_ln_rho_y = self.gradient_from_dec(np.log(self.compute_rho(psi1, psi2)))

        Ax = np.stack([ds1x-sx*d_ln_rho_x, ds2x-sy*d_ln_rho_x, ds3x-sz*d_ln_rho_x])             # riga x di A, shape (3, Nx, Ny)
        Ay = np.stack([ds1y-sx*d_ln_rho_y, ds2y-sy*d_ln_rho_y, ds3y-sz*d_ln_rho_y])             # riga y di A

        ux, uy = self.velocity_nodes(psi1, psi2)
        g2 = self.grad_psi_norm_squared(psi1, psi2)

        M1 = np.stack([ds1xx*sx + ds2xx*sy + ds3xx*sz, ds1xy*sx + ds2xy*sy + ds3xy*sz])  # shape (2, Nx, Ny)
        M2 = np.stack([ds1yx*sx + ds2yx*sy + ds3yx*sz, ds1yy*sx + ds2yy*sy + ds3yy*sz])  # shape (2, Nx, Ny)

        R1 = np.stack([d_rdrx_x, d_rdry_x])
        R2 = np.stack([d_rdrx_y, d_rdry_y])

        """
        # (8 m^2 mu / hbar^2) rho grad u   (hbar^2, non hbar)
        # ATTENZIONE convenzione degli indici: qui la riga x e' (d_x u_x, d_x u_y), quindi
        # S_x = sum_j d_j rho * d_x u_j = [(grad u) . grad rho]_x. Se nel paper grad rho . grad u
        # indica (grad rho . grad) u, la riga x va invece (d_x u_x, d_y u_x): da verificare.
        N1 = 8 * self.mu * self.compute_rho(psi1, psi2) / self.hbar**2 * np.stack([d_vx_x, d_vy_x])
        N2 = 8 * self.mu * self.compute_rho(psi1, psi2) / self.hbar**2 * np.stack([d_vx_y, d_vy_y])
        """

        coef = 8 * self.mu * self.compute_rho(psi1, psi2) / self.hbar**2
        N1 = coef * np.stack([d_vx_x, d_vx_y])   # S_x ⊃ d_x rho d_x u_x + d_y rho d_y u_x
        N2 = coef * np.stack([d_vy_x, d_vy_y])   # S_y ⊃ d_x rho d_x u_y + d_y rho d_y u_y

        Termine_nuovo1_x = 2 *self.mu / (self.hbar * self.compute_rho(psi1, psi2)) * g2 * ux
        Termine_nuovo1_y = 2 *self.mu / (self.hbar * self.compute_rho(psi1, psi2)) * g2 * uy

        Termine_nuovo2_x = self.mu / self.hbar * (divu + vx * d_ln_rho_x + vy * d_ln_rho_y) * d_ln_rho_x
        Termine_nuovo2_y = self.mu / self.hbar * (divu + vx * d_ln_rho_x + vy * d_ln_rho_y) * d_ln_rho_y

        gx = [self.gradient_from_dec(c) for c in (psi1, psi2)]          # d_k psi_c
        Mx = np.zeros(psi1.shape); My = np.zeros(psi1.shape)
        for (dx_, dy_) in gx:
            dxx, dxy = self.gradient_from_dec(dx_); dyx, dyy = self.gradient_from_dec(dy_)
            Mx += np.imag(np.conj(dx_)*dxx + np.conj(dy_)*dyx)
            My += np.imag(np.conj(dx_)*dxy + np.conj(dy_)*dyy)

        Sx = (0.25 * self.hbar / self.compute_rho(psi1,psi2)**2 * np.sum(Drho * (R1-M1 + N1), axis=0) 
                + Termine_nuovo1_x - Termine_nuovo2_x -2 * self.mu / self.compute_rho(psi1, psi2) * Mx)
        
        Sy = (0.25 * self.hbar / self.compute_rho(psi1,psi2)**2 * np.sum(Drho * (R2-M2 + N2), axis=0)
                + Termine_nuovo1_y - Termine_nuovo2_y - 2 * self.mu / self.compute_rho(psi1, psi2) * My)

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

    def compute_enthalpy(self, psi1, psi2):
        # int dp/rho per p = c^2 rho  ->  c^2 ln(rho)   (c fissato nel costruttore)
        return self.c**2 * np.log(self.compute_rho(psi1, psi2))

    def compute_P_quantum(self, psi1, psi2):
        """
        P_q tale che grad P_q = (1/rho) grad X, X = hbar^2 |grad rho|^2 / (4 rho).
        (1/rho) grad X in generale NON e' un gradiente: se ne prende la parte a gradiente
        con una decomposizione di Hodge discreta, sugli edge:
            G = dX / rho_edge,   Lap P_q = div G.
        last_Pq_solenoidal = |G - dP_q| / |G|  misura la parte non integrabile (il limite
        della mappatura).
        """
        rho = self.compute_rho(psi1, psi2)
        drx, dry = self.gradient_from_dec(rho)
        X = 0.25 * self.hbar**2 * (drx**2 + dry**2) / rho
        dXx, dXy = self.derivative_of_function(X)                 # 1-forma sugli edge
        rho_ex = 0.5 * (rho + np.roll(rho, -1, axis=0))           # rho sugli edge
        rho_ey = 0.5 * (rho + np.roll(rho, -1, axis=1))
        Gx, Gy = dXx / rho_ex, dXy / rho_ey
        Pq = self.poisson_solve(self.div_one_form(Gx, Gy))
        dPx, dPy = self.derivative_of_function(Pq)
        num = np.mean((Gx - dPx)**2 + (Gy - dPy)**2)
        den = np.mean(Gx**2 + Gy**2)
        self.last_Pq_solenoidal = np.sqrt(num / den) if den > 0 else 0.0
        return Pq

    def compute_mu_quantum(self, psi1, psi2):

        rho = self.compute_rho(psi1, psi2)
        ux, uy = self.spinor_velocity_one_form(psi1, psi2)
        #duxx, duxy = self.gradient_from_dec(ux)
        #duyx, duyy = self.gradient_from_dec(uy)
        X = self.div_one_form(ux,uy)
        dXx, dXy = self.derivative_of_function(X)                 # 1-forma sugli edge
        rho_ex = 0.5 * (rho + np.roll(rho, -1, axis=0))           # rho sugli edge
        rho_ey = 0.5 * (rho + np.roll(rho, -1, axis=1))
        Gx, Gy = self.mu *dXx / (3*rho_ex), self.mu *dXy / (3*rho_ey)
        mu_q = self.poisson_solve(self.div_one_form(Gx, Gy))
        dmuqx, dmuqy = self.derivative_of_function(mu_q)
        num = np.mean((Gx - dmuqx)**2 + (Gy - dmuqy)**2)
        den = np.mean(Gx**2 + Gy**2)
        self.last_muq_solenoidal = np.sqrt(num / den) if den > 0 else 0.0
        return mu_q

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
        # pressione fisica (solo diagnostica): p = c^2 rho
        return self.c**2 * self.compute_rho(psi1, psi2)