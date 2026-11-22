import numpy as np
from .ISF_DECbased import ISF2D_DECbased


class viscous_incompressible_SF(ISF2D_DECbased):
    """
    Fluido viscoso incomprimibile (Meng & Yang, Eq. 13 / tesi Eq. 5.38), m = rho0 = 1:

        i hbar dPsi/dt = [ (1 - 2i nu/hbar) p^2/2 + p_tilde + hbar^2/2 |grad Psi|^2
                           + i hbar nu |grad Psi|^2 ] Psi + sigma.B Psi

    Eredita da ISF2D_DECbased: normalize, pressure_project, advance (flow -> normalize ->
    project), operatori DEC e diagnostiche restano quelli del caso inviscido.
    Con nu = 0 deve riprodurre ISF2D_DECbased a precisione di macchina (test di regressione).

    Cosa cambia rispetto al caso inviscido:
      1. passo cinetico: la maschera acquista il fattore di smorzamento exp(nu*lam*dt);
      2. potenziale immaginario V_i = hbar*nu*|grad Psi|^2;
      3. sigma.B contiene f, soluzione di A f = S con A_ij = d_i s_j (2x3 in 2D)
         e S = 2 nu |grad Psi|^2 u / hbar  (phi_h = 0 perche' in 2D l'elicita' e' nulla).

    Termini che NON serve calcolare, perche' assorbiti esattamente dai vincoli:
      - p_tilde: potenziale scalare reale -> fase puntuale e^{-i p_tilde dt/hbar}
        -> cambia u di un gradiente -> rimosso esattamente da pressure_project;
      - V_i: fattore reale scalare e^{V_i dt/hbar} uguale sulle due componenti
        -> rimosso esattamente da normalize (puntuale).
      V_i viene comunque applicato (flag applica_vi) perche' rende fisicamente sensato lo
      stato prima della normalizzazione (utile per le diagnostiche su rho).
    """

    def __init__(self, sizex, sizey, resx, resy, nu, hbar, dt, eps_f=1e-8, applica_vi=True):
        super().__init__(sizex, sizey, resx, resy, hbar, dt)
        self.nu = nu
        self.eps_f = eps_f            # regolarizzazione di Tikhonov (relativa) per f
        self.applica_vi = applica_vi
        self.viscous_mask = None

    # ------------------------------------------------------------------ cinetica
    def build_schrodinger(self):
        # d psi/dt = (i hbar/2 + nu) Lap psi  ->  exp((i hbar/2 + nu) lam dt)
        super().build_schrodinger()
        nx, ny = self.resx, self.resy
        lam = (
            -4*np.sin(np.pi * self.iix / nx)**2 / self.dx**2
            -4*np.sin(np.pi * self.iiy / ny)**2 / self.dy**2
        )
        self.viscous_mask = np.exp(lam * self.nu * self.dt) * self.schrodinger_mask

    # ------------------------------------------------------------------ flusso
    def schrodinger_flow(self, psi1, psi2, correzione="correzione esatta"):
        if self.viscous_mask is None:
            raise ValueError("Call build_schrodinger() before schrodinger_flow().")

        psi1_star = np.fft.ifftn(np.fft.fftn(psi1) * self.viscous_mask)
        psi2_star = np.fft.ifftn(np.fft.fftn(psi2) * self.viscous_mask)

        if correzione == "correzione off":
            return psi1_star.copy(), psi2_star.copy()

        # potenziali congelati allo stato di inizio passo (come nel caso inviscido)
        vr = self.compute_vr(psi1, psi2)
        vi = self.compute_vi(psi1, psi2) if self.applica_vi else 0.0
        p, q = self.compute_p_q(psi1, psi2)

        dth = self.dt / self.hbar
        amp = np.exp(vi * dth)                        # parte immaginaria: fattore reale

        if correzione == "correzione on":
            # approssimazione al prim'ordine del paper (NB: qui con /hbar, coerente con l'esatta)
            phase1 = np.exp(-1j * (vr + p) * dth)
            phase2 = np.exp(-1j * (vr - p) * dth)
            coupling12 = np.exp(-1j * q * dth) - 1.0
            coupling21 = np.exp(-1j * np.conjugate(q) * dth) - 1.0
            psi1_new = amp * (psi1_star * phase1 + psi2_star * coupling12)
            psi2_new = amp * (psi2_star * phase2 + psi1_star * coupling21)

        elif correzione == "correzione esatta":
            beta = np.sqrt(p**2 + np.abs(q)**2)
            phase = amp * np.exp(-1j * vr * dth)
            argument = beta * dth
            c = np.cos(argument)
            s_over_beta = np.empty_like(beta)
            mask = beta > 1e-14
            s_over_beta[mask] = np.sin(argument[mask]) / beta[mask]
            s_over_beta[~mask] = dth
            psi1_new = phase * (c * psi1_star
                                - 1j * s_over_beta * (p * psi1_star + q * psi2_star))
            psi2_new = phase * (c * psi2_star
                                - 1j * s_over_beta * (np.conjugate(q) * psi1_star - p * psi2_star))
        else:
            raise ValueError("correzione must be 'correzione on', 'correzione off' o 'correzione esatta'.")
        return psi1_new, psi2_new

    # advance() e' ereditato: schrodinger_flow -> normalize -> pressure_project

    # ------------------------------------------------------------------ potenziali
    def compute_vi(self, psi1, psi2):
        # V_i = hbar * nu * |grad Psi|^2 / rho0   (NON hbar^2)
        return self.hbar * self.nu * self.grad_psi_norm_squared(psi1, psi2)

    def compute_p_q(self, psi1, psi2):
        p, q = super().compute_p_q(psi1, psi2)        # parte hbar^2/4 Lap s (inviscida)
        if self.nu == 0:
            return p, q                               # f = 0 esattamente
        f1, f2, f3 = self.compute_f(psi1, psi2)
        p = p - self.hbar * f3
        q = q - self.hbar * (f1 - 1j * f2)
        return p, q

    def velocity_nodes(self, psi1, psi2):
        # u ai nodi: la 1-forma vive sugli edge e vale u*dx -> sharp (divide per dx e media)
        return self.sharp_vortex(*self.spinor_velocity_one_form(psi1, psi2))

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
        Sx = 2 * self.nu * g2 * ux / self.hbar
        Sy = 2 * self.nu * g2 * uy / self.hbar

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

    # ------------------------------------------------------------------ diagnostica
    def compute_pressure(self, psi1, psi2):
        # pressura classica (solo diagnostica): Lap p = -d_i u_j d_j u_i
        ux, uy = self.velocity_nodes(psi1, psi2)
        du_dx, du_dy = self.gradient_from_dec(ux)
        dv_dx, dv_dy = self.gradient_from_dec(uy)
        return self.poisson_solve(-(du_dx**2 + 2 * du_dy * dv_dx + dv_dy**2))