import numpy as np
from .TorusDEC2D import TorusDEC2D


class inviscid_compressible_prova(TorusDEC2D):
    """
    Eulero comprimibile barotropico (tesi Eq. 5.44-5.46), m = 1:

        i hbar dPsi/dt = [ p^2/2 + p_tilde + hbar^2/(2 rho) |grad Psi|^2 ] Psi + sigma.B Psi

        p_tilde = h(rho) - P_q - |u|^2/2 + (hbar/rho) s.f
        h(rho)  = int dp/rho = c^2 ln(rho)              (isoterma, p = c^2 rho)
        grad P_q = (1/rho) grad X,  X = hbar^2 |grad rho|^2 / (4 rho)
        P = hbar^2 Lap(s3)/(4 rho) - hbar f3,  Q = hbar^2 (Lap s1 - i Lap s2)/(4 rho) - hbar (f1 - i f2)
        A_ij = d_i s_j - s_j d_i ln rho,   S = hbar/(4 rho^2) grad rho . [grad(rho grad rho) - grad grad s . s]

    Niente normalizzazione e niente proiezione: rho = |Psi|^2 e' una variabile del sistema.
    Il passo e' unitario (cinetico + locale), quindi la massa totale si conserva esattamente.
    """

    def __init__(self, sizex, sizey, resx, resy, hbar, Mach, dt, U_ref=1.0, eps_f=1e-8):
        super().__init__(sizex, sizey, resx, resy)
        self.hbar = hbar
        self.Mach = Mach
        self.U_ref = U_ref
        self.c = U_ref / Mach          # velocita' del suono FISSA (equazione di stato)
        self.dt = dt
        self.eps_f = eps_f
        self.schrodinger_mask = None
        self.last_f_residual = np.nan
        self.last_Pq_solenoidal = np.nan

    # ------------------------------------------------------------------ cinetica
    def build_schrodinger(self):
        nx, ny = self.resx, self.resy
        lam = (
            -4*np.sin(np.pi * self.iix / nx)**2 / self.dx**2
            -4*np.sin(np.pi * self.iiy / ny)**2 / self.dy**2
        )
        self.schrodinger_mask = np.exp(1j * lam * self.hbar * self.dt / 2)

    # ------------------------------------------------------------------ flusso
    def schrodinger_flow(self, psi1, psi2, correzione="correzione esatta"):
        if self.schrodinger_mask is None:
            raise ValueError("Call build_schrodinger() before schrodinger_flow().")

        psi1_star = np.fft.ifftn(np.fft.fftn(psi1) * self.schrodinger_mask)
        psi2_star = np.fft.ifftn(np.fft.fftn(psi2) * self.schrodinger_mask)

        if correzione == "correzione off":
            return psi1_star.copy(), psi2_star.copy()

        # potenziali congelati allo stato di inizio passo; f calcolato UNA volta
        f = self.compute_f(psi1, psi2)
        vr = self.compute_vr(psi1, psi2)
        p_tilde = self.compute_p_tilde(psi1, psi2, f)
        p, q = self.compute_p_q(psi1, psi2, f)

        dth = self.dt / self.hbar
        scal = np.exp(-1j * (vr + p_tilde) * dth)      # parte scalare (identita')

        if correzione == "correzione on":
            phase1 = np.exp(-1j * p * dth)
            phase2 = np.exp(1j * p * dth)
            coupling12 = np.exp(-1j * q * dth) - 1.0
            coupling21 = np.exp(-1j * np.conjugate(q) * dth) - 1.0
            psi1_new = scal * (psi1_star * phase1 + psi2_star * coupling12)
            psi2_new = scal * (psi2_star * phase2 + psi1_star * coupling21)

        elif correzione == "correzione esatta":
            beta = np.sqrt(p**2 + np.abs(q)**2)
            argument = beta * dth
            c = np.cos(argument)
            s_over_beta = np.empty_like(beta)
            mask = beta > 1e-14
            s_over_beta[mask] = np.sin(argument[mask]) / beta[mask]
            s_over_beta[~mask] = dth
            psi1_new = scal * (c * psi1_star
                               - 1j * s_over_beta * (p * psi1_star + q * psi2_star))
            psi2_new = scal * (c * psi2_star
                               - 1j * s_over_beta * (np.conjugate(q) * psi1_star - p * psi2_star))
        else:
            raise ValueError("correzione must be 'correzione on', 'correzione off' o 'correzione esatta'.")
        return psi1_new, psi2_new

    def advance(self, psi1, psi2, correzione="correzione esatta"):
        # comprimibile: niente normalize, niente pressure_project
        return self.schrodinger_flow(psi1, psi2, correzione=correzione)

    # ------------------------------------------------------------------ potenziali
    def compute_rho(self, psi1, psi2):
        return np.abs(psi1)**2 + np.abs(psi2)**2

    def compute_vr(self, psi1, psi2):
        # hbar^2 |grad Psi|^2 / (2 rho)
        return self.hbar**2 * self.grad_psi_norm_squared(psi1, psi2) / (2 * self.compute_rho(psi1, psi2))

    def compute_p_q(self, psi1, psi2, f=None):
        sx, sy, sz = self.hopf(psi1, psi2)
        rho = self.compute_rho(psi1, psi2)
        f1, f2, f3 = self.compute_f(psi1, psi2) if f is None else f
        k = 0.25 * self.hbar**2 / rho
        p = k * self.laplacian_from_dec(sz) - self.hbar * f3
        q = k * (self.laplacian_from_dec(sx) - 1j * self.laplacian_from_dec(sy)) \
            - self.hbar * (f1 - 1j * f2)
        return p, q

    def compute_enthalpy(self, psi1, psi2):
        # int dp/rho per p = c^2 rho  ->  c^2 ln(rho)   (c fissato nel costruttore)
        return self.c**2 * np.log(self.compute_rho(psi1, psi2))

    def compute_pressure(self, psi1, psi2):
        # pressione fisica (solo diagnostica)
        return self.c**2 * self.compute_rho(psi1, psi2)

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

    def compute_p_tilde(self, psi1, psi2, f=None):
        rho = self.compute_rho(psi1, psi2)
        ux, uy = self.velocity_nodes(psi1, psi2)
        sx, sy, sz = self.hopf(psi1, psi2)
        f1, f2, f3 = self.compute_f(psi1, psi2) if f is None else f
        return (self.compute_enthalpy(psi1, psi2)
                - self.compute_P_quantum(psi1, psi2)
                - 0.5 * (ux**2 + uy**2)
                + self.hbar / rho * (sx * f1 + sy * f2 + sz * f3))

    def compute_f(self, psi1, psi2):
        """
        A f = S punto per punto, A (2x3): A_ij = d_i s_j - s_j d_i ln(rho);
        S = hbar/(4 rho^2) grad rho . [grad(rho grad rho) - grad grad s . s]  (tensore simmetrico).
        Pseudo-inversa di Tikhonov: f = A^T (A A^T + eps I)^{-1} S.
        """
        sx, sy, sz = self.hopf(psi1, psi2)
        rho = self.compute_rho(psi1, psi2)
        drx, dry = self.gradient_from_dec(rho)
        dlx, dly = self.gradient_from_dec(np.log(rho))

        grads = [self.gradient_from_dec(s) for s in (sx, sy, sz)]       # (d_x s_j, d_y s_j)
        Ax = np.stack([g[0] - s * dlx for g, s in zip(grads, (sx, sy, sz))])
        Ay = np.stack([g[1] - s * dly for g, s in zip(grads, (sx, sy, sz))])

        # tensore T_jk = d_j(rho d_k rho) - sum_l s_l d_j d_k s_l
        Txx_r, Txy_r = self.gradient_from_dec(rho * drx)          # d_x(rho d_x rho), d_y(rho d_x rho)
        Tyx_r, Tyy_r = self.gradient_from_dec(rho * dry)          # d_x(rho d_y rho), d_y(rho d_y rho)
        Mxx = Mxy = Myy = 0.0
        for (gx, gy), s in zip(grads, (sx, sy, sz)):
            hxx, hxy = self.gradient_from_dec(gx)
            hyx, hyy = self.gradient_from_dec(gy)
            Mxx = Mxx + s * hxx
            Mxy = Mxy + s * 0.5 * (hxy + hyx)                     # simmetrizzato
            Myy = Myy + s * hyy
        Txx = Txx_r - Mxx
        Txy = 0.5 * (Txy_r + Tyx_r) - Mxy
        Tyy = Tyy_r - Myy

        pref = 0.25 * self.hbar / rho**2
        Sx = pref * (drx * Txx + dry * Txy)
        Sy = pref * (drx * Txy + dry * Tyy)

        a = np.sum(Ax * Ax, axis=0)
        b = np.sum(Ax * Ay, axis=0)
        d = np.sum(Ay * Ay, axis=0)
        eps = self.eps_f * np.max(a + d)
        a, d = a + eps, d + eps
        det = a * d - b**2
        y1 = ( d * Sx - b * Sy) / det
        y2 = (-b * Sx + a * Sy) / det
        f = Ax * y1 + Ay * y2

        rx = np.sum(Ax * f, axis=0) - Sx
        ry = np.sum(Ay * f, axis=0) - Sy
        self.last_f_residual = np.sqrt(np.mean(rx**2 + ry**2) /
                                       max(np.mean(Sx**2 + Sy**2), 1e-300))
        return f[0], f[1], f[2]

    # ------------------------------------------------------------------ operatori
    def spinor_velocity_one_form(self, psi1, psi2):
        # arg(psi^dag psi(x+dx)) ~ Im(psi^dag d psi)/rho * dx : corretto anche con rho non uniforme
        vx = np.angle(np.conjugate(psi1) * np.roll(psi1, -1, axis=0) +
                      np.conjugate(psi2) * np.roll(psi2, -1, axis=0))
        vy = np.angle(np.conjugate(psi1) * np.roll(psi1, -1, axis=1) +
                      np.conjugate(psi2) * np.roll(psi2, -1, axis=1))
        return vx * self.hbar, vy * self.hbar

    def velocity_nodes(self, psi1, psi2):
        return self.sharp_vortex(*self.spinor_velocity_one_form(psi1, psi2))

    def vorticity_two_form(self, vx, vy):
        return self.derivative_of_one_form(vx, vy)

    def grad_psi_norm_squared(self, psi1, psi2):
        d1x, d1y = self.gradient_from_dec(psi1)
        d2x, d2y = self.gradient_from_dec(psi2)
        return np.abs(d1x)**2 + np.abs(d1y)**2 + np.abs(d2x)**2 + np.abs(d2y)**2

    def gradient_from_dec(self, f):
        return self.sharp_vortex(*self.derivative_of_function(f))

    def laplacian_from_dec(self, f):
        return self.div_one_form(*self.derivative_of_function(f))

    @staticmethod
    def hopf(psi1, psi2):
        # s non normalizzato: |s| = rho
        a, b = np.real(psi1), np.imag(psi1)
        c, d = np.real(psi2), np.imag(psi2)
        return 2 * (a * c + b * d), 2 * (a * d - b * c), a**2 + b**2 - c**2 - d**2