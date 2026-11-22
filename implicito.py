"""
implicito.py -- trattamento implicito del potenziale locale (caso inviscido incomprimibile)

Schemi confrontati (tutti con passo cinetico esatto + normalize + pressure_project):
  "frozen"  : Lie con potenziale congelato a t^n              (isf.advance, "correzione esatta")
  "updated" : predittore-correttore, media dei potenziali      (= Lie_updated di Davide)
  "picard"  : regola dei trapezi implicita sul potenziale, risolta per punto fisso fino a convergenza
  "nk"      : stessa equazione implicita, risolta con Newton-Krylov (Jacobian-free)

Equazione implicita (incognita = stato FINALE, gia' normalizzato e proiettato):
    psi^{n+1} = G(psi^{n+1}),
    G(psi)    = P o N [ U( (V(psi^n) + V(psi)) / 2 ) K psi^n ]
con K passo cinetico, U passo locale esatto (2x2), N normalize, P pressure_project.
Il potenziale e' quindi sempre valutato su uno stato normalizzato e proiettato.

Uso nel notebook:
    from metodi2Dperiodic.implicito import *
    energy = lambda isf, a, b: diagnostics_from_state(isf, a, b)["E"]
    ris = confronto_dt(ISF2D_DECbased, make_TG, energy, vol_size, hbar,
                       resolution=80, dt_list=[...], T=1.0)
    plot_confronto(ris)
    rapp = rapporti_contrazione(ISF2D_DECbased, make_TG, vol_size, hbar, 80, dt_list)
    plot_contrazione(rapp)
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton_krylov
from scipy.optimize.nonlin import NoConvergence


class SchemaFallito(Exception):
    """Il passo implicito non converge (o supera il tempo massimo)."""


# --------------------------------------------------------------------------
# Mattoni: identici al codice di Davide
# --------------------------------------------------------------------------
def potenziali(isf, psi):
    vr = isf.compute_vr(*psi)
    p, q = isf.compute_p_q(*psi)
    return vr, p, q


def applica_V(isf, vr, p, q, psi):
    beta = np.sqrt(p**2 + np.abs(q)**2)
    phase = np.exp(-1j * vr * isf.dt / isf.hbar)
    arg = beta * isf.dt / isf.hbar
    s_ob = np.zeros_like(beta, dtype=np.complex128)
    m = beta > 1e-14
    s_ob[m] = np.sin(arg[m]) / beta[m]
    s_ob[~m] = isf.dt / isf.hbar
    a, b = psi
    c = np.cos(arg)
    return (phase * (c * a - 1j * s_ob * (p * a + q * b)),
            phase * (c * b - 1j * s_ob * (np.conjugate(q) * a - p * b)))


def passo_cinetico(isf, psi1, psi2):
    return (np.fft.ifftn(np.fft.fftn(psi1) * isf.schrodinger_mask),
            np.fft.ifftn(np.fft.fftn(psi2) * isf.schrodinger_mask))


def advance_lie_updated(isf, psi1, psi2):
    V_n = potenziali(isf, (psi1, psi2))
    k = passo_cinetico(isf, psi1, psi2)
    pred = applica_V(isf, *V_n, k)
    V_n1 = potenziali(isf, pred)
    V_avg = tuple(0.5 * (a + b) for a, b in zip(V_n, V_n1))
    out = applica_V(isf, *V_avg, k)
    out = isf.normalize(*out)
    return isf.pressure_project(*out)


# --------------------------------------------------------------------------
# Equazione implicita
# --------------------------------------------------------------------------
def make_G(isf, p1n, p2n):
    """Mappa G del punto fisso psi = G(psi) per il passo n -> n+1."""
    V_n = potenziali(isf, (p1n, p2n))
    k = passo_cinetico(isf, p1n, p2n)

    def G(psi):
        V_avg = tuple(0.5 * (a + b) for a, b in zip(V_n, potenziali(isf, psi)))
        out = isf.normalize(*applica_V(isf, *V_avg, k))
        return isf.pressure_project(*out)
    return G


# Il potenziale dipende da psi e psi* -> F non e' olomorfa -> si lavora in variabili reali
def pack(p1, p2):
    return np.concatenate([p1.real.ravel(), p1.imag.ravel(),
                           p2.real.ravel(), p2.imag.ravel()])


def unpack(X, shape):
    r = X.reshape(4, -1)
    return (r[0] + 1j * r[1]).reshape(shape), (r[2] + 1j * r[3]).reshape(shape)


def advance_picard(isf, psi1, psi2, stats, tol=1e-10, maxiter=300):
    """Punto fisso fino a convergenza (norma max dell'incremento < tol)."""
    G = make_G(isf, psi1, psi2)
    psi = advance_lie_updated(isf, psi1, psi2)          # guess iniziale
    d_min = np.inf
    for it in range(1, maxiter + 1):
        new = G(psi)
        d = max(np.abs(new[0] - psi[0]).max(), np.abs(new[1] - psi[1]).max())
        psi = new
        if d < tol:
            stats["iter"].append(it)
            return psi
        if d > 10 * d_min:                  # l'incremento esplode: il punto fisso non contrae
            raise SchemaFallito(f"punto fisso divergente (it={it})")
        d_min = min(d_min, d)
    raise SchemaFallito(f"punto fisso: maxiter={maxiter} raggiunto")


def advance_nk(isf, psi1, psi2, stats, f_tol=1e-10, maxiter=50, inner_maxiter=100,
               t_limite=None):
    """Newton-Krylov (LGMRES, Jacobiano approssimato con differenze finite)."""
    G = make_G(isf, psi1, psi2)
    sh = psi1.shape
    cont = [0]

    def F(X):
        cont[0] += 1
        if t_limite is not None and time.time() > t_limite:
            raise TimeoutError
        return X - pack(*G(unpack(X, sh)))

    X0 = pack(*advance_lie_updated(isf, psi1, psi2))    # guess iniziale
    try:
        X = newton_krylov(F, X0, f_tol=f_tol, maxiter=maxiter,
                          inner_maxiter=inner_maxiter, method="lgmres")
    except NoConvergence:
        raise SchemaFallito(f"Newton-Krylov non converge ({cont[0]} valutazioni di F)")
    stats["iter"].append(cont[0])                       # costo = valutazioni di F
    return unpack(X, sh)


# --------------------------------------------------------------------------
# Esperimenti
# --------------------------------------------------------------------------
SCHEMI = ("frozen", "updated", "picard", "nk")


def run_schema(ISFclass, make_TG, energy, vol_size, hbar, resolution, dt, T, schema,
               t_max_run=600.0, soglia_div=1.0):
    """Un run fino a T. Restituisce un dict con esito, errore d'energia e costo."""
    isf = ISFclass(vol_size[0], vol_size[1], resolution, resolution, hbar=hbar, dt=dt)
    isf.build_schrodinger()
    p1, p2 = make_TG(isf)
    E0 = energy(isf, p1, p2)
    nsteps = int(np.ceil(T / dt))
    stats = {"iter": []}
    t0 = time.time()
    esito, n = "ok", 0
    try:
        for n in range(nsteps):
            if schema == "frozen":
                p1, p2 = isf.advance(p1, p2, correzione="correzione esatta")
            elif schema == "updated":
                p1, p2 = advance_lie_updated(isf, p1, p2)
            elif schema == "picard":
                p1, p2 = advance_picard(isf, p1, p2, stats)
            elif schema == "nk":
                p1, p2 = advance_nk(isf, p1, p2, stats, t_limite=t0 + t_max_run)
            else:
                raise ValueError(schema)
            if time.time() - t0 > t_max_run:
                esito = "tempo"
                break
    except SchemaFallito as e:
        esito = f"non converge: {e}"
    except TimeoutError:
        esito = "tempo"

    err = np.nan
    if esito == "ok":
        err = abs(energy(isf, p1, p2) / E0 - 1.0)
        if not np.isfinite(err) or err > soglia_div:
            esito = "diverge"
    return {"schema": schema, "dt": dt, "err": err, "esito": esito,
            "passi": n + 1, "passi_tot": nsteps,
            "costo_medio": np.mean(stats["iter"]) if stats["iter"] else np.nan,
            "tempo": time.time() - t0}


def confronto_dt(ISFclass, make_TG, energy, vol_size, hbar, resolution, dt_list, T=1.0,
                 schemi=SCHEMI, t_max_run=600.0):
    ris = {s: [] for s in schemi}
    cfl = min(vol_size) ** 2 / resolution ** 2 / (2 * hbar)
    print(f"res={resolution}²   soglia CFL dx²/(2ħ) = {cfl:.2e}")
    for dt in dt_list:
        for s in schemi:
            r = run_schema(ISFclass, make_TG, energy, vol_size, hbar, resolution, dt, T, s,
                           t_max_run=t_max_run)
            ris[s].append(r)
            costo = "" if np.isnan(r["costo_medio"]) else \
                f"  iter/passo={r['costo_medio']:.1f}"
            print(f"[{s:8}] dt={dt:.2e}  |E/E0-1|={r['err']:.3e}  {r['esito']:<12}"
                  f"  passi={r['passi']}/{r['passi_tot']}  t={r['tempo']:.1f}s{costo}",
                  flush=True)
        print()
    ris["_meta"] = {"resolution": resolution, "T": T, "cfl": cfl}
    return ris


def plot_confronto(ris, stili=None):
    meta = ris["_meta"]
    stili = stili or {"frozen": "o-", "updated": "d-", "picard": "s-", "nk": "^-"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    for s, lista in ris.items():
        if s == "_meta":
            continue
        dt = np.array([r["dt"] for r in lista])
        err = np.array([r["err"] for r in lista])
        ok = np.array([r["esito"] == "ok" for r in lista])
        linea, = ax1.loglog(dt[ok], err[ok], stili.get(s, "o-"), lw=2, label=s)
        if (~ok).any():   # fallimenti: croci sul bordo superiore
            ax1.plot(dt[~ok], np.full((~ok).sum(), 1.0), "x", ms=11, mew=2.5,
                     color=linea.get_color())
        costo = np.array([r["costo_medio"] for r in lista])
        if np.isfinite(costo).any():
            ax2.semilogx(dt[ok], costo[ok], stili.get(s, "o-"), lw=2, label=s,
                         color=linea.get_color())
    for ax in (ax1, ax2):
        ax.axvline(meta["cfl"], ls=":", color="k", label=r"$\Delta x^2/(2\hbar)$")
        ax.grid(alpha=0.3, which="both")
        ax.set_xlabel(r"$\Delta t$")
    ax1.set_ylabel(r"$|E(T)/E_0-1|$   (× = diverge / non converge / tempo)")
    ax1.set_title(f"Errore d'energia a T={meta['T']}, res={meta['resolution']}")
    ax2.set_ylabel("iterazioni per passo (picard) / valutazioni di F (nk)")
    ax2.set_title("Costo del passo implicito")
    ax1.legend(); ax2.legend()
    plt.tight_layout()
    plt.show()


def rapporti_contrazione(ISFclass, make_TG, vol_size, hbar, resolution, dt_list, n_it=30):
    """Rapporto ||d_{k+1}|| / ||d_k|| del punto fisso al primo passo, per ogni dt.
    Il rapporto cresce lentamente con k (emerge l'autovalore dominante): servono
    ~30 iterazioni; con poche iterazioni la contrazione viene sottostimata."""
    out = []
    for dt in dt_list:
        isf = ISFclass(vol_size[0], vol_size[1], resolution, resolution, hbar=hbar, dt=dt)
        isf.build_schrodinger()
        p1, p2 = make_TG(isf)
        G = make_G(isf, p1, p2)
        psi = advance_lie_updated(isf, p1, p2)
        d_prev, rapporti = None, []
        for _ in range(n_it):
            new = G(psi)
            d = np.sqrt(np.sum(np.abs(new[0] - psi[0])**2 + np.abs(new[1] - psi[1])**2))
            if d < 1e-12 or d > 1e3:        # convergenza a macchina o divergenza netta: stop
                break
            if d_prev:
                rapporti.append(d / d_prev)
            d_prev, psi = d, new
        r = np.exp(np.mean(np.log(rapporti[-3:])))   # media geometrica degli ultimi rapporti
        out.append((dt, r))
        print(f"dt={dt:.2e}  rapporto di contrazione ~ {r:.3f}")
    cfl = min(vol_size) ** 2 / resolution ** 2 / (2 * hbar)
    return {"dt": np.array([o[0] for o in out]), "r": np.array([o[1] for o in out]),
            "cfl": cfl, "resolution": resolution}


def plot_contrazione(rapp):
    plt.figure(figsize=(8, 5.5))
    plt.plot(rapp["dt"], rapp["r"], "o-", lw=2, label="rapporto di contrazione")
    plt.axhline(1.0, color="r", ls="--", label="limite del punto fisso")
    plt.axvline(rapp["cfl"], color="k", ls=":", label=r"$\Delta x^2/(2\hbar)$")
    plt.xlabel(r"$\Delta t$")
    plt.ylabel(r"$\|\delta_{k+1}\| / \|\delta_k\|$")
    plt.title(f"Contrazione del punto fisso al primo passo, res={rapp['resolution']}")
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()