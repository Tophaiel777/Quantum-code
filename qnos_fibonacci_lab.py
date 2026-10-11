#!/usr/bin/env python3
"""
QNOS FIBONACCI LAB -- fractal / Fibonacci frequency modulation of chiral phonons, quantum scars and birthmarks
===============================================================================================================
Requires: numpy, scipy, matplotlib   (scikit-image optional: used for marching cubes, a built-in mesher otherwise)
Self-contained: it does NOT need qnos_physics20262.py (the models are re-implemented here, same equations).

    python qnos_fibonacci_lab.py "any text you like"
    python qnos_fibonacci_lab.py "any text" --out results --quick       # fast pass (coarser grids, fewer shuffles)
    python qnos_fibonacci_lab.py "any text" --show                      # also open the dashboard window
    python qnos_fibonacci_lab.py                                        # interactive prompt

WHAT IT DOES
------------
1. The text is hashed (SHA-256) and fixes ONE physical configuration (same scheme as qnos_proofs.py):
   honeycomb antiferromagnet (masses, springs, spin-phonon coupling g, hidden Neel sign) + stadium wavepacket.
2. A library of SIX frequency modulations is built. Every drive is  E(t) = E0 env(t) [cos th, pol sin th]  with
       th(t) = w0 ( t + m J(t) ),   J(t) = integral of the shape s(t),   |s| <= 1,   m = peak fractional detuning.
       cw          s = 0                                           (reference)
       chirp       s = linear sweep                                (periodic-style reference of equal depth)
       fibword     s from the Fibonacci word (A->AB, B->A), 55 slices, A = +1/phi, B = -1 (zero mean)
       golden_qp   s = [cos(W t) + cos(phi W t)]/2                 (two incommensurate tones, ratio = golden mean)
       fractal_wm  s = Weierstrass-Mandelbrot sum, scale ratio phi, 8 octaves, Hurst H = 0.7
       cantor      s = 0, drive gated on the level-4 Cantor set    (fractal on/off pulse train)
3. PHONON LAB: the circularly polarised pulse drives the q=0 optical doublet. Measured: absorbed energy for
   sigma+ / sigma-, dichroic contrast C = (A+ - A-)/(A+ + A-), decoded Neel sign, vs modulation depth and vs carrier
   frequency (dichroism spectrum). Exact energy balance dE/dt = F.v - gamma v.M.v is integrated alongside the ODE.
4. SCAR LAB: stadium eigenstates (sparse shift-invert). The same modulations drive the packet through a dipole
   coupling  H(t) = H0 + eps f(t) x/(Ls+R)  projected on the eigenbasis (exactly unitary Strang splitting).
   Measured: heating, memory of the launch (birthmark enhancement B, window-matched against the undriven packet),
   fractal (box-counting) dimension of the birthmark ridges, and a BIRTHMARK ATLAS on a Fibonacci/Vogel (golden-angle)
   lattice of 55 launch points with a paired sign-flip test: does Fibonacci FM heat differently from an equal-depth chirp?
5. AMDUAT-MRI STYLE VOLUMES: (a) phonon worldline tubes, x=u_x, y=u_y, z=time, for sigma+ and sigma-;
   (b) scar volume, x,y = billiard, z = modulation depth. Both exported as .obj (marching cubes) and rendered in 3D.
6. VERIFICATION CERTIFICATE (text + JSON) with IDENTITY / CLOSED-FORM / BOUND / ASYMPTOTIC / DYNAMICS / STATISTICAL /
   CONTROL claims and an auto-written FINDINGS section built only from the measured numbers.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
A classical-computer simulation of MODEL Hamiltonians (minimal lattice dynamics; a billiard with a projected dipole
drive). It needs no quantum computer because every object here is a small linear / unitary problem that is solved
exactly or to stated tolerance. It is not an ab-initio material calculation and not a formal proof; "PASS" means a
numerical claim held at the text-selected parameter point to the stated tolerance. Findings are measurements of the model.

REFERENCES (real papers; read them for the actual theory)
  E. J. Heller, PRL 53, 1515 (1984) scars;  L. Zhang & Q. Niu, PRL 115, 115502 (2015) chiral phonons;
  H. Ueda et al., Nature 618, 946 (2023) chiral phonons by circular-dichroic RIXS;
  Quantum birthmarks, arXiv:2412.02982;
  P. T. Dumitrescu et al., Nature 607, 463 (2022) Fibonacci (quasiperiodic) driving and dynamical protection.
"""
import argparse, hashlib, json, math, os, platform, sys, time

import numpy as np
import matplotlib
if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy
import scipy.sparse as sp
from scipy import ndimage, signal
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import eigsh
try:
    from skimage import measure
    HAVE_SK = True
except ImportError:
    HAVE_SK = False

VERSION = "1.0"
PHI = (1 + 5 ** 0.5) / 2
GOLDEN_ANGLE = 2 * math.pi / PHI ** 2
S3 = math.sqrt(3.0)
BG = "#05080c"
WAVES = ["cw", "chirp", "fibword", "golden_qp", "fractal_wm", "cantor"]
WNAME = {"cw": "CW (reference)", "chirp": "linear chirp", "fibword": "Fibonacci word FM",
         "golden_qp": "golden quasi-periodic FM", "fractal_wm": "fractal Weierstrass FM", "cantor": "Cantor-gated"}
COL = {"cw": "#9aa7b5", "chirp": "#ffd166", "fibword": "#00f0ff", "golden_qp": "#ff40d0",
       "fractal_wm": "#7CFC00", "cantor": "#ff8c42"}
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": "#dfe9f0",
                     "axes.labelcolor": "#7fd", "xtick.color": "#8a9aa5", "ytick.color": "#8a9aa5",
                     "axes.edgecolor": "#35505a", "axes.titlecolor": "#0ff", "axes.titlesize": 9,
                     "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
                     "legend.facecolor": BG, "legend.edgecolor": "#35505a", "legend.labelcolor": "#dfe9f0"})


# =============================================================================
# 1. FIBONACCI / FRACTAL MODULATION LIBRARY
# =============================================================================
def fib_word(n):
    """Fibonacci word after n substitution steps (A->AB, B->A), starting from 'A'. Length = F_{n+2}."""
    w = "A"
    for _ in range(n):
        w = "".join("AB" if ch == "A" else "A" for ch in w)
    return w


def cantor_mask(x, level):
    """True where x in [0,1] lies in the level-`level` Cantor set approximation (middle thirds removed)."""
    x = np.clip(np.asarray(x, float), 0.0, 1.0 - 1e-12)
    keep = np.ones(x.shape, bool)
    for _ in range(level):
        x3 = x * 3.0
        d = np.floor(x3)
        keep &= (d != 1)
        x = x3 - d
    return keep


def make_tables(t, T, seed=0, fib_n=9, cantor_level=4, H=0.7, n_oct=8):
    """Return (J, env, rms) arrays of shape (len(t), 6): J = integral of the detuning shape, env = pulse envelope."""
    dt = t[1] - t[0]
    x = np.clip(t / T, 0.0, 1.0)
    base_env = np.where((t >= 0) & (t <= T), np.sin(np.pi * np.clip(t, 0, T) / T) ** 2, 0.0)
    word = fib_word(fib_n)
    vals = np.array([1.0 / PHI if ch == "A" else -1.0 for ch in word])
    idx = np.minimum((x * len(word)).astype(int), len(word) - 1)
    W1 = 2 * np.pi * 5 / T
    psi = np.random.default_rng(seed).uniform(0, 2 * np.pi, n_oct)
    wm = sum(PHI ** (-H * n) * np.cos(PHI ** n * 2 * np.pi * 2 / T * t + psi[n]) for n in range(n_oct))
    wm = wm / np.max(np.abs(wm))
    gate = cantor_mask(x, cantor_level).astype(float)
    kw = max(3, int(T / 3 ** cantor_level / 3 / dt))
    ker = np.hanning(kw + 2)[1:-1]
    gate = np.convolve(gate, ker / ker.sum(), mode="same")
    shapes = {"cw": np.zeros_like(t), "chirp": 2 * x - 1, "fibword": vals[idx],
              "golden_qp": 0.5 * (np.cos(W1 * t) + np.cos(PHI * W1 * t)), "fractal_wm": wm, "cantor": np.zeros_like(t)}
    J = np.empty((len(t), len(WAVES)))
    E = np.empty_like(J)
    rms = np.empty(len(WAVES))
    for i, w in enumerate(WAVES):
        s = shapes[w]
        J[:, i] = np.concatenate([[0.0], np.cumsum(0.5 * (s[1:] + s[:-1]) * np.diff(t))])
        E[:, i] = base_env * (gate if w == "cantor" else 1.0)
        rms[i] = math.sqrt(np.sum(s ** 2 * base_env) / max(np.sum(base_env), 1e-30))
    return J, E, rms


def box_dimension(mask, sizes=(2, 3, 4, 6, 8, 12, 16)):
    counts = []
    for s in sizes:
        ny, nx = (mask.shape[0] // s) * s, (mask.shape[1] // s) * s
        blk = mask[:ny, :nx].reshape(ny // s, s, nx // s, s).any(axis=(1, 3))
        counts.append(max(int(blk.sum()), 1))
    return float(-np.polyfit(np.log(sizes), np.log(counts), 1)[0])


# =============================================================================
# 2. SEEDED CONFIGURATION
# =============================================================================
class SeededUniverse:
    def __init__(self, text, R=1.0, Ls=1.0):
        self.text = text
        self.digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        self.seed = int(self.digest, 16)
        r = np.random.default_rng(np.random.SeedSequence(self.seed))
        U = lambda a, b: float(r.uniform(a, b))
        self.mA = 1.0
        self.mB = U(1.6, 3.0)
        self.phi_l = U(0.8, 1.4)
        self.phi_t = self.phi_l * U(0.25, 0.5)
        self.g = U(0.25, 0.45)
        self.neel = int(r.choice([-1, 1]))
        self.kappa = 1.5 * (self.phi_l + self.phi_t)
        self.k0, self.sigma, self.angle = U(15.0, 20.0), U(0.14, 0.22), U(0.0, 360.0)
        for _ in range(100000):
            x, y = U(-1.7, 1.7), U(-0.7, 0.7)
            if R - math.hypot(x - min(max(x, -Ls), Ls), y) > 3 * self.sigma:
                break
        self.launch = (x, y)
        ss = lambda k: np.random.default_rng(np.random.SeedSequence([self.seed & (2 ** 128 - 1), k]))
        self.rng_k, self.rng_perm, self.rng_cal, self.rng_flip = ss(1), ss(2), ss(3), ss(4)
        self.wave_seed = int(self.seed & 0xFFFFFFFF)

    def params(self):
        return {k: getattr(self, k) for k in ("mA", "mB", "phi_l", "phi_t", "g", "neel", "k0", "sigma", "angle", "launch")}


RESULTS = []


def rec(rid, group, title, kind, claim, measured, tol, passed, note="", status=None, mode="max"):
    r = dict(id=rid, group=group, title=title, kind=kind, claim=claim, measured=float(measured), tol=float(tol),
             mode=mode, status=status or ("PASS" if passed else "FAIL"), note=note)
    RESULTS.append(r)
    return r


# =============================================================================
# 3. PHONON MODEL  (honeycomb AFM with spin-phonon gyroscopic term; same equations as QNOS physics)
# =============================================================================
class PhononModel:
    a1, a2 = np.array([1.0, 0.0]), np.array([0.5, S3 / 2])
    b1, b2 = 2 * np.pi * np.array([1, -1 / S3]), 2 * np.pi * np.array([0, 2 / S3])
    deltas = np.array([[0.5, 1 / (2 * S3)], [-0.5, 1 / (2 * S3)], [0.0, -1 / S3]])

    def __init__(self, mA=1.0, mB=1.8, phi_l=1.0, phi_t=0.35, g=0.30, neel=+1):
        self.mA, self.mB, self.g, self.neel = mA, mB, g, neel
        self.M = np.diag([mA, mA, mB, mB])
        self.Minv = np.linalg.inv(self.M)
        self.Ks = []
        for d in self.deltas:
            u = d / np.linalg.norm(d)
            P = np.outer(u, u)
            self.Ks.append(phi_l * P + phi_t * (np.eye(2) - P))
        self.S = sum(self.Ks)
        J = np.array([[0.0, 1.0], [-1.0, 0.0]])
        G = np.zeros((4, 4))
        G[:2, :2] = g * neel * J
        G[2:, 2:] = -g * neel * J
        self.G = G

    def phi(self, k):
        ab = -sum(K * np.exp(1j * k @ d) for K, d in zip(self.Ks, self.deltas))
        return np.block([[self.S, ab], [ab.conj().T, self.S]])

    def solve(self, k):
        Z, I4 = np.zeros((4, 4)), np.eye(4)
        A = np.block([[Z, I4], [-self.Minv @ self.phi(k), self.Minv @ self.G]])
        lam, V = np.linalg.eig(A)
        om = 1j * lam
        idx = np.argsort(-om.real)[:4]
        idx = idx[np.argsort(om.real[idx])]
        w, eps = om.real[idx], V[:4, idx]
        ell = np.empty(4)
        for j in range(4):
            e = eps[:, j].reshape(2, 2)
            ell[j] = 2 * np.sum(np.array([self.mA, self.mB]) * np.imag(np.conj(e[:, 0]) * e[:, 1])) / \
                np.sum(np.array([self.mA, self.mB]) * np.sum(np.abs(e) ** 2, axis=1))
        return w, eps, ell


def mk(U, s, g=None, mB=None):
    return PhononModel(mA=U.mA, mB=U.mB if mB is None else mB, phi_l=U.phi_l, phi_t=U.phi_t,
                       g=U.g if g is None else g, neel=s)


def closed_form_gamma(U, g=None, mB=None, s=None):
    g = U.g if g is None else g
    mA, mB = U.mA, (U.mB if mB is None else mB)
    s = U.neel if s is None else s
    D = math.sqrt(g * g * (mA - mB) ** 2 + 4 * mA * mB * (U.kappa * (mA + mB) + g * g))
    return dict(ccw=(g * s * (mA - mB) + D) / (2 * mA * mB), cw=(-g * s * (mA - mB) + D) / (2 * mA * mB))


# ---- vectorised batch RK4 for the driven q=0 problem, with exact energy bookkeeping ---------------------------
def phonon_grid(T=160.0, dt=0.04, tail=1.35):
    n = int(round(tail * T / dt))
    return np.arange(2 * n + 1) * (dt / 2.0), dt, n


def drive_batch(model, pol, om0, m, wi, tabs, E0=0.02, gamma=0.03, dt=0.04, rec_every=0):
    """Integrate nc independent drives at once. State = [u(4), v(4), W_in, W_diss]. tabs=(t, J, env)."""
    t, Jm, Em = tabs
    om0, m, wi = np.atleast_1d(om0).astype(float), np.atleast_1d(m).astype(float), np.atleast_1d(wi).astype(int)
    nc = max(len(om0), len(m), len(wi), np.size(pol))
    om0, m, wi = (np.broadcast_to(a, (nc,)) for a in (om0, m, wi))
    pol = np.broadcast_to(np.asarray(pol, float), (nc,))
    n = (len(t) - 1) // 2
    th = om0[None, :] * (t[:, None] + m[None, :] * Jm[:, wi])
    env = Em[:, wi]
    Ex = np.ascontiguousarray(E0 * env * np.cos(th))
    Ey = np.ascontiguousarray(E0 * env * pol[None, :] * np.sin(th))
    del th, env
    Phi0 = model.phi(np.zeros(2)).real
    Minv, mdiag = model.Minv, np.diag(model.M)
    Aint = np.zeros((8, 8))
    Aint[:4, 4:] = np.eye(4)
    Aint[4:, :4] = -Minv @ Phi0
    Aint[4:, 4:] = Minv @ (model.G - gamma * model.M)
    AT, MiT = Aint.T.copy(), Minv.T.copy()

    def rhs(y, ex, ey):
        d = np.empty_like(y)
        d[:, :8] = y[:, :8] @ AT
        F = np.stack((ex, ey, -ex, -ey), axis=1)
        d[:, 4:8] += F @ MiT
        v = y[:, 4:8]
        d[:, 8] = np.einsum("ij,ij->i", F, v)
        d[:, 9] = gamma * (v * v * mdiag).sum(1)
        return d

    y = np.zeros((nc, 10))
    amp = np.zeros(nc)
    rec = []
    for i in range(n):
        a, b, c = 2 * i, 2 * i + 1, 2 * i + 2
        k1 = rhs(y, Ex[a], Ey[a])
        k2 = rhs(y + 0.5 * dt * k1, Ex[b], Ey[b])
        k3 = rhs(y + 0.5 * dt * k2, Ex[b], Ey[b])
        k4 = rhs(y + dt * k3, Ex[c], Ey[c])
        y = y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        amp = np.maximum(amp, np.maximum(np.hypot(y[:, 0], y[:, 1]), np.hypot(y[:, 2], y[:, 3])))
        if rec_every and (i + 1) % rec_every == 0:
            rec.append(y[:, :4].copy())
    u, v = y[:, :4], y[:, 4:8]
    Eend = 0.5 * (v * v * mdiag).sum(1) + 0.5 * np.einsum("ij,jk,ik->i", u, Phi0, u)
    return dict(Win=y[:, 8], Wd=y[:, 9], Eend=Eend, amp=amp, rec=np.array(rec) if rec_every else None)


def drive_reference(model, pol, om0, m, wi, tabs, E0=0.02, gamma=0.03):
    """Independent integration of the same drive with scipy DOP853 (for the integrator-agreement claim)."""
    t, Jm, Em = tabs
    Phi0 = model.phi(np.zeros(2)).real
    Z = np.array([1, 1, -1, -1.0])

    def f(tt, y):
        th = om0 * (tt + m * np.interp(tt, t, Jm[:, wi]))
        e = E0 * np.interp(tt, t, Em[:, wi])
        F = Z * np.tile([e * math.cos(th), e * pol * math.sin(th)], 2)
        u, v = y[:4], y[4:]
        return np.concatenate([v, model.Minv @ (-Phi0 @ u + model.G @ v - gamma * model.M @ v + F)])
    sol = solve_ivp(f, (0, t[-1]), np.zeros(8), t_eval=t[50::50], method="DOP853", rtol=1e-10, atol=1e-13, max_step=0.05)
    return sol.y[:4].T


# =============================================================================
# 4. STADIUM BILLIARD + DIPOLE-DRIVEN WAVEPACKET
# =============================================================================
class Stadium:
    def __init__(self, h=0.03, R=1.0, Ls=1.0):
        self.h, self.R, self.Ls = h, R, Ls
        self.xs = np.arange(-(Ls + R) + h / 2, Ls + R, h)
        self.ys = np.arange(-R + h / 2, R, h)
        self.X, self.Y = np.meshgrid(self.xs, self.ys)
        self.mask = (np.maximum(np.abs(self.X) - Ls, 0) ** 2 + self.Y ** 2) < R ** 2
        self.idx = -np.ones(self.mask.shape, int)
        self.idx[self.mask] = np.arange(self.mask.sum())
        self.N = int(self.mask.sum())

    def hamiltonian(self):
        h, idx, N = self.h, self.idx, self.N
        rows, cols, vals = [np.arange(N)], [np.arange(N)], [np.full(N, 2.0 / h ** 2)]
        for dy, dx in ((0, 1), (1, 0)):
            a = idx[:idx.shape[0] - dy, :idx.shape[1] - dx]
            b = idx[dy:, dx:]
            ok = (a >= 0) & (b >= 0)
            for p, q in ((a[ok], b[ok]), (b[ok], a[ok])):
                rows.append(p); cols.append(q); vals.append(np.full(p.size, -0.5 / h ** 2))
        return sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))

    def to_grid(self, vec):
        g = np.full(self.mask.shape, np.nan)
        g[self.mask] = vec
        return g


class ScarLab:
    def __init__(self, h, nev, k0, eps_scale=1.0):
        self.st = Stadium(h)
        self.H = self.st.hamiltonian()
        self.nev = nev
        E, V = eigsh(self.H, k=nev, sigma=k0 ** 2 / 2, which="LM")
        o = np.argsort(E)
        self.E, self.V = E[o], V[:, o]
        self.V2 = self.V ** 2
        self.ref = self.V2.mean(axis=1)
        self.X = self.st.X[self.st.mask]
        self.Y = self.st.Y[self.st.mask]
        self.D = self.V.T @ ((self.X / (self.st.Ls + self.st.R))[:, None] * self.V)     # dipole operator in eigenbasis
        self.lam, self.Uv = np.linalg.eigh(self.D)

    def packet(self, x0, y0, k0, ang_deg, sigma):
        a = math.radians(ang_deg)
        X, Y = self.X, self.Y
        psi = np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (4 * sigma ** 2)) * \
            np.exp(1j * (k0 * math.cos(a) * (X - x0) + k0 * math.sin(a) * (Y - y0)))
        psi /= np.linalg.norm(psi)
        return self.V.T @ psi

    def disk(self, x0, y0, r=0.3):
        return np.where(np.hypot(self.X - x0, self.Y - y0) < r)[0]

    def carrier(self, c0, lo=1.5, hi=16.0):
        w = np.linspace(lo, hi, 300)
        S = self.dipole_spectrum(c0, w)
        return float(w[np.argmax(S)]), w, S

    def dipole_spectrum(self, c0, w, width=0.4):
        p = np.abs(c0) ** 2
        dE = (self.E[None, :] - self.E[:, None]).ravel()
        W = (self.D ** 2 * p[:, None]).ravel()
        return np.array([np.sum(W * np.exp(-(dE - x) ** 2 / (2 * width ** 2))) for x in w])

    def evolve(self, C0, F, dt, sample=10, disks=None, want_rho=False, chunk=48):
        """Strang-split evolution of nc packets. C0 (nev,nc); F (n,nc) = coupling strength at step mid-points."""
        nc = F.shape[1]
        C0 = np.broadcast_to(C0, (self.nev, nc)) if C0.ndim == 2 and C0.shape[1] == nc else np.tile(C0.reshape(-1, 1), (1, nc))
        Cf = np.empty((self.nev, nc), complex)
        B = np.full(nc, np.nan)
        rho = np.empty((nc, self.st.N)) if want_rho else None
        ph = np.exp(-0.5j * self.E * dt)[:, None]
        for s0 in range(0, nc, chunk):
            sl = slice(s0, min(nc, s0 + chunk))
            c = np.array(C0[:, sl], complex)
            Fc = F[:, sl]
            cb = np.zeros((c.shape[1], self.nev, self.nev), complex)
            ns = 0
            for i in range(F.shape[0]):
                c = ph * c
                c = self.Uv @ (np.exp(-1j * dt * self.lam[:, None] * Fc[i][None, :]) * (self.Uv.T @ c))
                c = ph * c
                if i % sample == 0:
                    cb += np.einsum("ik,jk->kij", c, c.conj(), optimize=True)
                    ns += 1
            cb /= ns
            Cf[:, sl] = c
            for k in range(c.shape[1]):
                gk = s0 + k
                tr = cb[k].trace().real
                if disks is not None:
                    d = disks[gk]
                    Vd = self.V[d]
                    B[gk] = (((Vd @ cb[k]) * Vd).sum(1).real / tr).mean() / self.ref[d].mean()
                if want_rho:
                    rho[gk] = ((self.V @ cb[k]) * self.V).sum(1).real / tr
        return dict(Cf=Cf, B=B, rho=rho)

    def energy(self, C):
        p = np.abs(C) ** 2
        return (p * self.E[:, None]).sum(0) / p.sum(0)


def scar_tables(T, dt, seed):
    n = int(round(T / dt))
    t = (np.arange(n) + 0.5) * dt
    J, E, rms = make_tables(t, T, seed=seed)
    return t, J, E, rms


def scar_force(stab, wi, om0, m, eps):
    t, J, E, _ = stab
    wi = np.atleast_1d(wi).astype(int)
    om0 = np.broadcast_to(np.atleast_1d(om0).astype(float), wi.shape)
    m = np.broadcast_to(np.atleast_1d(m).astype(float), wi.shape)
    th = om0[None, :] * (t[:, None] + m[None, :] * J[:, wi])
    return eps * E[:, wi] * np.cos(th)


def vogel_launches(sigma, n=55, seed_angle=0.0):
    pts = []
    k = 1
    while len(pts) < n and k < 600:
        r, th = math.sqrt((k - 0.5) / 112.0), k * GOLDEN_ANGLE + seed_angle
        x, y = 1.95 * r * math.cos(th), 0.95 * r * math.sin(th)
        cx = min(max(x, -1.0), 1.0)
        if 1.0 - math.hypot(x - cx, y) > 3 * sigma:
            pts.append((x, y, math.degrees((k * GOLDEN_ANGLE) % (2 * math.pi))))
        k += 1
    return pts


# =============================================================================
# 5. MESHES / VOLUMES  (Amduat-MRI idea: stack of slices -> scalar field -> isosurface -> .obj)
# =============================================================================
def voxel_mesher(vol, level):
    P = np.pad(vol > level, 1)
    quads = []
    off = np.array([(-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5)])
    for ax in range(3):
        o = [a for a in range(3) if a != ax]
        for sgn in (-1, 1):
            nb = np.roll(P, -sgn, axis=ax)
            idx = np.argwhere(P & ~nb) - 1
            for q in idx:
                c = q.astype(float) + 0.5
                c[ax] += 0.5 * sgn
                corners = []
                for a_, b_ in off:
                    p = c.copy(); p[o[0]] += a_; p[o[1]] += b_; corners.append(p)
                quads.append(corners if sgn > 0 else corners[::-1])
    if not quads:
        return np.zeros((0, 3)), np.zeros((0, 3), int)
    verts = np.array(quads).reshape(-1, 3)
    base = np.arange(len(quads)) * 4
    faces = np.concatenate([np.stack([base, base + 1, base + 2], 1), np.stack([base, base + 2, base + 3], 1)])
    return verts, faces


def mesh_from_volume(vol, level, spacing=(1.0, 1.0, 1.0)):
    if vol.max() <= level or vol.min() >= level:
        return np.zeros((0, 3)), np.zeros((0, 3), int)
    if HAVE_SK:
        v, f, _, _ = measure.marching_cubes(vol, level=level, spacing=spacing)
        return v, f
    v, f = voxel_mesher(vol, level)
    return v * np.array(spacing), f


def write_obj(path, verts, faces, header):
    with open(path, "w") as fh:
        fh.write(f"# {header}\n")
        for v in verts:
            fh.write(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n")
        for f in faces:
            fh.write(f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}\n")


def phonon_volumes(model, om0, m, wave_i, tabs, res=64, depth=96, periods=8):
    r = drive_batch(model, np.array([+1.0, -1.0]), om0, m, wave_i, tabs, rec_every=1)
    t = tabs[0][::2][1:]
    path = r["rec"]                                              # (nstep, 2, 4)
    uB = np.hypot(path[:, :, 2], path[:, :, 3])
    i0 = int(np.argmax(uB[:, 0]))
    half = int(periods * 2 * math.pi / om0 / 0.04 / 2)
    a, b = max(0, i0 - half), min(len(t), i0 + half)
    umax = np.abs(path[a:b, :, 2:]).max() + 1e-30
    vols, curves = [], []
    for k in range(2):
        npt = 10 * (b - a)
        src = np.linspace(0, 1, b - a)
        dst = np.linspace(0, 1, npt)
        x = np.interp(dst, src, path[a:b, k, 2]) / umax * 0.42 * res + res / 2
        y = np.interp(dst, src, path[a:b, k, 3]) / umax * 0.42 * res + res / 2
        z = dst * (depth - 1)
        vol = np.zeros((res, res, depth))
        np.add.at(vol, (np.clip(np.rint(x).astype(int), 0, res - 1), np.clip(np.rint(y).astype(int), 0, res - 1),
                        np.rint(z).astype(int)), 1.0)
        vol = ndimage.gaussian_filter(vol, 1.5)
        vols.append(vol / vol.max())
        curves.append((x, y, z))
    return vols, curves


def draw_mesh(ax, verts, faces, cmap, zcol, shift=(0, 0, 0), maxf=7000, alpha=0.95, seed=1):
    if len(faces) == 0:
        return
    if len(faces) > maxf:
        faces = faces[np.random.default_rng(seed).choice(len(faces), maxf, replace=False)]
    tri = verts[faces] + np.array(shift)
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-30
    shade = 0.35 + 0.65 * np.abs(nrm @ np.array([0.4, 0.3, 0.86]))
    zc = tri[:, :, zcol].mean(1)
    zc = (zc - zc.min()) / (np.ptp(zc) + 1e-30)
    fc = cmap(0.1 + 0.85 * zc)
    fc[:, :3] *= shade[:, None]
    fc[:, 3] = alpha
    pc = Poly3DCollection(tri, facecolors=fc, edgecolors="none", linewidths=0)
    ax.add_collection3d(pc)


def style3d(ax, title, lim):
    ax.set_facecolor(BG)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_pane_color((0.02, 0.04, 0.06, 1.0))
        a.set_tick_params(colors="#566", labelsize=6)
    ax.set_title(title, color="#0ff", fontsize=9)
    ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1]); ax.set_zlim(*lim[2])
    ax.set_box_aspect((lim[0][1] - lim[0][0], lim[1][1] - lim[1][0], lim[2][1] - lim[2][0]), zoom=1.25)


# =============================================================================
# 6. PHONON CLAIMS AND EXPERIMENTS
# =============================================================================
def phonon_claims(U, ptab, a):
    s = U.neel
    u, v = U.rng_k.random((2, 16))
    ks = u[:, None] * PhononModel.b1 + v[:, None] * PhononModel.b2
    mp_, mm_ = mk(U, +1), mk(U, -1)
    ms = mp_ if s > 0 else mm_
    herm = max(np.abs(ms.phi(k) - ms.phi(k).conj().T).max() for k in ks)
    real = max(np.abs(ms.phi(-k) - ms.phi(k).conj()).max() for k in ks)
    psd = max(0.0, -min(np.linalg.eigvalsh(ms.phi(k)).min() for k in ks))
    anti = np.abs(ms.G + ms.G.T).max()
    e = max(herm, real, psd, anti)
    rec("P1", "phonon", "Force-matrix structure", "IDENTITY",
        "Phi(k) Hermitian, Phi(-k)=conj Phi(k), Phi >= 0, G real antisymmetric.", e, 1e-12, e <= 1e-12)
    Z, I4 = np.zeros((4, 4)), np.eye(4)
    worst = max(np.abs(np.linalg.eigvals(np.block([[Z, I4], [-ms.Minv @ ms.phi(k), ms.Minv @ ms.G]])).real).max() for k in ks)
    rec("P2", "phonon", "Dynamical stability", "IDENTITY", "all w real (max |Re lambda| = 0).", worst, 1e-8, worst <= 1e-8)
    tr = max(np.abs(mp_.solve(k)[0] - mm_.solve(-k)[0]).max() for k in ks)
    rec("P3", "phonon", "Time-reversal relation (D2)", "IDENTITY", "w_n(k, s) = w_n(-k, -s).", tr, 1e-9, tr <= 1e-9)
    nr = max(np.abs(ms.solve(k)[0] - ms.solve(-k)[0]).max() for k in ks)
    m0 = mk(U, s, g=0.0)
    nr0 = max(np.abs(m0.solve(k)[0] - m0.solve(-k)[0]).max() for k in ks)
    rec("P4", "phonon", "Non-reciprocity and its g=0 null", "CONTROL", "max_k |w(k)-w(-k)| > 0 at g>0, exactly 0 at g=0.",
        nr, 1e-3, nr > 1e-3 and nr0 <= 1e-10, note=f"effect={nr:.4g}, null={nr0:.2e}", mode="min")

    def gam(m_):
        w, _, l = m_.solve(np.zeros(2))
        up = [j for j in (2, 3) if l[j] > 0]
        dn = [j for j in (2, 3) if l[j] < 0]
        return None if len(up) != 1 or len(dn) != 1 else dict(ccw=w[up[0]], cw=w[dn[0]], l=(l[up[0]], l[dn[0]]))
    ng, cf = gam(ms), closed_form_gamma(U)
    if ng is None:
        rec("G1", "phonon", "Gamma optical doublet (D1)", "CLOSED-FORM", "one CCW and one CW optical mode", np.inf, 1e-9, False)
    else:
        err = max(abs(ng["ccw"] - cf["ccw"]), abs(ng["cw"] - cf["cw"]), abs(abs(ng["l"][0]) - 1), abs(abs(ng["l"][1]) - 1))
        fl = closed_form_gamma(U, s=-s)
        sep = max(abs(ng["ccw"] - fl["ccw"]), abs(ng["cw"] - fl["cw"]))
        rec("G1", "phonon", "Gamma optical doublet vs analytic formula (D1)", "CLOSED-FORM",
            "w_CCW,CW = (+/- g s (mA-mB) + D)/(2 mA mB); modes purely circular.", err, 1e-9, err <= 1e-9 and sep > 1e-3,
            note=f"CCW={cf['ccw']:.8f} CW={cf['cw']:.8f}; wrong-sign formula would miss by {sep:.3g}")
        no = gam(mm_ if s > 0 else mp_)
        d1, d2 = ng["ccw"] - ng["cw"], no["ccw"] - no["cw"]
        ex = U.g * s * (1 / U.mB - 1 / U.mA)
        e2 = max(abs(d1 + d2), abs(d1 - ex))
        rec("G2", "phonon", "Splitting odd in Neel order, linear in g (D1)", "CLOSED-FORM",
            "w_CCW - w_CW = g s (1/mB - 1/mA).", e2, 1e-9, e2 <= 1e-9, note=f"splitting(s)={d1:+.8f} formula={ex:+.8f}")
    me = mk(U, s, mB=U.mA)
    wq = me.solve(np.zeros(2))[0]
    rec("G3", "phonon", "Inversion asymmetry required (mA = mB null)", "CLOSED-FORM",
        "With mA = mB the Gamma doublet stays degenerate.", abs(wq[2] - wq[3]), 1e-8, abs(wq[2] - wq[3]) <= 1e-8)

    # ---- modulation-aware dynamical claims
    t, dt, n = ptab["grid"]
    tabs = ptab["tabs"]
    om = closed_form_gamma(U, s=+1)["ccw"]
    model = mk(U, s)
    mdep = 0.08
    wi_all = np.arange(len(WAVES))
    # N1 integrator vs DOP853
    k = WAVES.index("fibword")
    rb = drive_batch(model, +1.0, om, mdep, k, tabs, rec_every=25)
    ref = drive_reference(model, +1.0, om, mdep, k, tabs)
    nmin = min(len(ref), len(rb["rec"]))
    err = np.abs(rb["rec"][:nmin, 0, :] - ref[:nmin]).max() / np.abs(ref).max()
    rec("N1", "phonon", "Batch RK4 agrees with scipy DOP853 (Fibonacci-word drive)", "DYNAMICS",
        "Two independent integrators give the same trajectory (limit set by table interpolation in the reference).", err, 1e-5, err <= 1e-5)
    # E1 energy balance, all waveforms
    r = drive_batch(model, +1.0, om, mdep, wi_all, tabs)
    bal = np.abs(r["Win"] - r["Wd"] - r["Eend"]) / np.abs(r["Win"])
    rec("E1", "phonon", "Power theorem: work in = dissipated + stored (six modulations)", "DYNAMICS",
        "W_in - W_diss - E_final = 0 (gyroscopic force does no work); residual is RK4 error, largest at the Cantor gate edges.", bal.max(), 1e-5, bal.max() <= 1e-5,
        note="worst " + WAVES[int(np.argmax(bal))])
    # M1 mirror covariance under every modulation
    rp = drive_batch(mk(U, s), +1.0, om, mdep, wi_all, tabs)
    rm = drive_batch(mk(U, -s), -1.0, om, mdep, wi_all, tabs)
    mir = max(np.abs(rp["Win"] - rm["Win"]).max() / rp["Win"].max(), np.abs(rp["amp"] - rm["amp"]).max() / rp["amp"].max())
    rec("M1", "phonon", "Mirror covariance holds for every modulation (D4)", "DYNAMICS",
        "Absorbed energy and amplitude of (pump, s) equal those of (-pump, -s) for ANY phase th(t).", mir, 1e-9, mir <= 1e-9)
    # L1 linearity
    r2 = drive_batch(model, +1.0, om, mdep, wi_all, tabs, E0=0.04)
    lin = max(np.abs(r2["amp"] / r["amp"] - 2).max(), np.abs(r2["Win"] / r["Win"] - 4).max())
    rec("L1", "phonon", "Linear response under modulation", "DYNAMICS", "E0 x2 => amplitude x2, absorbed energy x4.",
        lin, 1e-9, lin <= 1e-9)
    # R1 CW readout of the hidden order
    out = []
    ok = True
    for sg in (+1, -1):
        Ap = drive_batch(mk(U, sg), +1.0, om, 0.0, 0, tabs)["Win"][0]
        Am = drive_batch(mk(U, sg), -1.0, om, 0.0, 0, tabs)["Win"][0]
        dec = +1 if Ap > Am else -1
        ratio = max(Ap, Am) / min(Ap, Am)
        ok &= (dec == sg and ratio >= 1.5)
        out.append(f"s={sg:+d}: decoded {dec:+d}, dichroic ratio {ratio:.2f}")
    delta = abs(U.g * (1 / U.mB - 1 / U.mA))
    rec("R1", "phonon", "Hidden-order readout, CW reference pump", "DYNAMICS",
        "Matched helicity absorbs more; decoded sign equals the Neel order (ratio >= 1.5).", float(ok), 1.0, ok,
        status=None if delta >= 0.08 else "N/A", note="; ".join(out) + f"; seeded order {s:+d}; Gamma splitting {delta:.3f}", mode="min")
    # C1 g = 0 null for the dichroism under all modulations
    m00 = mk(U, s, g=0.0)
    a1 = drive_batch(m00, +1.0, om, mdep, wi_all, tabs)["Win"]
    a2 = drive_batch(m00, -1.0, om, mdep, wi_all, tabs)["Win"]
    c0 = np.abs((a1 - a2) / (a1 + a2)).max()
    rec("C1", "phonon", "Null control: no spin-phonon coupling => no dichroism", "CONTROL",
        "At g=0 sigma+ and sigma- are absorbed equally for every modulation.", c0, 1e-9, c0 <= 1e-9)
    # C2 negative control: broken mirror symmetry must be detected in the time-resolved amplitude
    bad = {}
    for pol, sg in ((+1.0, s), (-1.0, -s)):
        mm = mk(U, sg)
        mm.Ks[0] = mm.Ks[0] * 1.3
        mm.S = sum(mm.Ks)
        rr = drive_batch(mm, pol, om, mdep, k, tabs, rec_every=5)["rec"][:, 0, :]
        bad[pol] = np.maximum(np.hypot(rr[:, 0], rr[:, 1]), np.hypot(rr[:, 2], rr[:, 3]))
    brk = np.abs(bad[1.0] - bad[-1.0]).max() / max(bad[1.0].max(), bad[-1.0].max())
    rec("C2", "phonon", "Negative control: broken mirror symmetry", "CONTROL",
        "One bond stiffened by 30%: the time-resolved M1 comparison must now detect a violation (> 1e-3).", brk, 1e-3, brk > 1e-3,
        note=f"violation {brk:.3g}", mode="min")
    return om


def modulation_claims():
    # F1 Fibonacci word counting identity
    n = 12
    w = fib_word(n)
    F = [1, 1]
    while len(F) < n + 4:
        F.append(F[-1] + F[-2])
    okF = (len(w) == F[n + 1] and w.count("A") == F[n] and w.count("B") == F[n - 1])
    ratio_err = abs(w.count("A") / len(w) - 1 / PHI)
    rec("F1", "modulation", "Fibonacci word: length F(n+2), counts F(n+1) A and F(n) B", "IDENTITY",
        "Substitution A->AB, B->A gives exact Fibonacci counts; letter density -> 1/phi.", float(not okF), 0.0, okF,
        note=f"n={n}: len {len(w)}, A {w.count('A')}, B {w.count('B')}; density error vs 1/phi = {ratio_err:.2e}")
    # F2 bounded discrepancy (Sturmian)
    isA = np.array([ch == "A" for ch in w], float)
    disc = np.abs(np.cumsum(isA) - np.arange(1, len(w) + 1) / PHI).max()
    rec("F2", "modulation", "Fibonacci word has bounded discrepancy (quasi-periodic, not random)", "BOUND",
        "| #A(first N letters) - N/phi | < 1 for every prefix.", disc, 1.0, disc < 1.0, note=f"max discrepancy {disc:.4f} over {len(w)} prefixes")
    # F3 Cantor box dimension
    L = 3 ** 8
    msk = cantor_mask((np.arange(L) + 0.5) / L, 8)
    cnt = [int(msk[:L // 3 ** j * 3 ** j].reshape(-1, L // 3 ** j).any(1).sum()) for j in range(1, 8)]
    dim = np.polyfit(np.log([3.0 ** j for j in range(1, 8)]), np.log(cnt), 1)[0]
    e = abs(dim - math.log(2) / math.log(3))
    rec("F3", "modulation", "Cantor gate has box dimension ln2/ln3", "ASYMPTOTIC",
        "Box counting of the gate recovers the exact fractal dimension 0.6309.", e, 1e-6, e <= 1e-6, note=f"measured {dim:.6f}")
    # F4 golden-ratio incommensurability of the two-tone drive: autocorrelation never returns exactly
    t = np.arange(0, 4000, 0.05)
    s = 0.5 * (np.cos(t) + np.cos(PHI * t))
    ac_peak = np.max(np.abs(np.correlate(s[:20000], s[:20000], "full")[20000 + 200:])) / np.sum(s[:20000] ** 2)
    rec("F4", "modulation", "Golden two-tone drive is not periodic", "BOUND",
        "Normalised autocorrelation at lags > 10 stays below 1 (a periodic drive would return to 1).", ac_peak, 0.999, ac_peak < 0.999,
        note=f"max |autocorr| at lag>10: {ac_peak:.4f}")


# =============================================================================
# 7. SCAR CLAIMS AND EXPERIMENTS
# =============================================================================
def lattice_weyl(E, h, A, P, n=1500):
    k = (np.arange(n) + 0.5) / n * (math.pi / h)
    KX, KY = np.meshgrid(k, k)
    El = (2 - np.cos(KX * h) - np.cos(KY * h)) / h ** 2
    return A / (4 * math.pi ** 2) * 4 * np.mean(El < E) * (math.pi / h) ** 2 - P * math.sqrt(2 * E) / (4 * math.pi)


def scar_claims(U, lab, a, stab, w0, dts):
    st, E, V, nev = lab.st, lab.E, lab.V, lab.nev
    N = st.N
    res = np.max(np.linalg.norm(lab.H @ V - V * E, axis=0) / np.abs(E))
    orth = np.abs(V.T @ V - np.eye(nev)).max()
    rec("S1", "stadium", "Eigenpairs solve the Dirichlet problem", "IDENTITY", "||H psi - E psi||/E ~ 0, psi orthonormal.",
        max(res, orth), 1e-9, max(res, orth) <= 1e-9, note=f"residual {res:.2e}, orthonormality {orth:.2e}, {N} cells, {nev} states")
    A, P = math.pi * st.R ** 2 + 4 * st.R * st.Ls, 2 * math.pi * st.R + 4 * st.Ls
    pred = lattice_weyl(E[-1], st.h, A, P) - lattice_weyl(E[0], st.h, A, P)
    rel = abs(pred - nev) / nev
    rec("S2", "stadium", "Weyl law for the level count", "ASYMPTOTIC", "Counted levels match the lattice Weyl law.",
        rel, 0.05, rel <= 0.05, note=f"counted {nev}, predicted {pred:.1f}")
    c = lab.packet(*U.launch, U.k0, U.angle, U.sigma)
    p_raw = np.abs(c) ** 2
    cap = p_raw.sum()
    rec("S3", "stadium", "Bessel inequality / basis adequacy", "BOUND", "Captured packet weight <= 1.", cap, 1 + 1e-12,
        cap <= 1 + 1e-12, note=f"captured weight {cap:.4f}")
    # S4 driven unitarity, six modulations
    t, J, Em, rms = stab
    dt = a.sdt
    wi = np.arange(len(WAVES))
    F = scar_force(stab, wi, w0, a.mscar, a.eps)
    out = lab.evolve(c, F, dt, sample=1000000)
    drift = np.abs((np.abs(out["Cf"]) ** 2).sum(0) - cap).max() / cap
    rec("S4", "stadium", "Unitary driven evolution (six modulations)", "IDENTITY",
        "||psi(t)||^2 conserved under every Fibonacci / fractal drive.", drift, 1e-10, drift <= 1e-10)
    # S5 Strang convergence order (CW, analytic phase)
    Ts = a.sT
    refs = []
    for d in dts:
        n = int(round(Ts / d))
        tt = (np.arange(n) + 0.5) * d
        Fd = (a.eps * np.sin(np.pi * tt / Ts) ** 2 * np.cos(w0 * tt))[:, None]
        refs.append(lab.evolve(c, Fd, d, sample=1000000)["Cf"][:, 0])
    e1, e2 = np.linalg.norm(refs[0] - refs[2]), np.linalg.norm(refs[1] - refs[2])
    order = math.log2(e1 / max(e2, 1e-300))
    rec("S5", "stadium", "Strang splitting is second order", "ASYMPTOTIC", "Halving dt cuts the error ~4x (order ~2).",
        order, 1.7, order >= 1.7, note=f"dt={dts[0]:.4g},{dts[1]:.4g} vs {dts[2]:.4g}: errors {e1:.2e}, {e2:.2e}, order {order:.2f}", mode="min")
    # S6 first-order response: a COHERENT packet changes <H0> at first order in eps (not eps^2)
    Fw = lambda eps_: (eps_ * np.sin(np.pi * t / a.sT) ** 2 * np.cos(w0 * t))[:, None]
    dE = []
    for eps_ in (0.02, 0.04):
        o = lab.evolve(c, Fw(eps_), dt, sample=1000000)
        dE.append(lab.energy(o["Cf"])[0] - lab.energy(c.reshape(-1, 1))[0])
    ratio = dE[1] / dE[0]
    rec("S6", "stadium", "Weak-drive energy response is linear in eps (coherent packet)", "DYNAMICS",
        "Doubling eps doubles dE: a packet is a superposition, so <H0> responds at first order (eps^2 would apply to an eigenstate).",
        abs(ratio - 2) / 2, 0.03, abs(ratio - 2) / 2 <= 0.03,
        note=f"dE(0.02)={dE[0]:.3e}, dE(0.04)={dE[1]:.3e}, ratio {ratio:.3f} (linear = 2, quadratic = 4)")
    # S7 diagonal-ensemble bound (D5) and S8 birthmark test
    p = p_raw / cap
    V2 = lab.V2
    rho = V2 @ p
    dE_ = E[:, None] - E[None, :]
    iu = ~np.eye(nev, dtype=bool)
    cc = np.abs(c)[:, None] * np.abs(c)[None, :]
    Bs = np.sum(cc[iu] / np.abs(dE_[iu]))

    def avg_density(T_):
        with np.errstate(divide="ignore", invalid="ignore"):
            S = (1 - np.exp(-1j * dE_ * T_)) / (1j * dE_ * T_)
        S[~iu] = 1.0
        C = (c[:, None] * np.conj(c)[None, :]) * S
        return np.real(np.sum((V @ C) * V, axis=1)) / cap
    Tl = np.array([1, 3, 10, 30, 100, 300, 1000.0])
    errs = np.array([np.abs(avg_density(T_) - rho).sum() for T_ in Tl])
    bounds = 2 * Bs / Tl / cap
    rec("S7", "stadium", "Convergence to the diagonal ensemble (D5)", "BOUND",
        "|| <rho>_T - rho_bar ||_1 <= (2/T) sum |c_n c_m|/|E_n-E_m|.", float(np.max(errs / bounds)), 1.0,
        bool(np.all(errs <= bounds * (1 + 1e-9))), note=f"max error/bound {np.max(errs / bounds):.3f}")
    ref = lab.ref
    disk = lab.disk(*U.launch)
    stratum = (np.arange(nev) // 10).astype(float)
    refd = ref[disk].mean()

    def stats(pw, rg, n_):
        Bl, Tl_ = [], []
        for _ in range(max(1, n_ // 500)):
            idx = np.argsort(stratum[:, None] + rg.random((nev, 500)) * 0.999, axis=0)
            rn = V2[disk] @ pw[idx]
            Bl.append(rn.mean(axis=0) / refd)
            Tl_.append(0.5 * np.abs(V2 @ pw[idx] - ref[:, None]).sum(axis=0))
        return np.concatenate(Bl), np.concatenate(Tl_)
    B_obs, tv_obs = rho[disk].mean() / refd, 0.5 * np.abs(rho - ref).sum()
    nB, nT = stats(p, U.rng_perm, a.shuffles)
    pB, pT = (1 + np.sum(nB >= B_obs)) / (1 + nB.size), (1 + np.sum(nT >= tv_obs)) / (1 + nT.size)
    pmin = min(pB, pT)
    det = pmin < 0.005
    rec("S8", "stadium", "Birthmark: memory of the launch (undriven)", "STATISTICAL",
        "H0: within energy strata packet weights are unrelated to eigenstate structure; Bonferroni alpha=0.01.", pmin, 0.005, det,
        status="DETECTED" if det else "NOT DETECTED",
        note=f"B={B_obs:.2f} (null {nB.mean():.2f}+/-{nB.std():.2f}, z={(B_obs - nB.mean()) / (nB.std() + 1e-300):.1f}, p={pB:.4f}); "
             f"TV={tv_obs:.3f} (null {nT.mean():.3f}, p={pT:.4f}); {nB.size} stratified shuffles")
    n_cal, fp = 24, 0
    for _ in range(n_cal):
        pseudo = p[np.argsort(stratum + U.rng_cal.random(nev) * 0.999)]
        b_ps, t_ps = V2[disk] @ pseudo, 0.5 * np.abs(V2 @ pseudo - ref).sum()
        cb, ct = stats(p, U.rng_cal, 500)
        pv = min((1 + np.sum(cb >= b_ps.mean() / refd)) / (1 + cb.size), (1 + np.sum(ct >= t_ps)) / (1 + ct.size))
        fp += pv < 0.025
    rec("S9", "stadium", "Calibration of the birthmark test (type-I error)", "CONTROL",
        "Under a true null the false-positive rate at alpha=0.05 stays <= 25%.", fp / n_cal, 0.25, fp / n_cal <= 0.25,
        note=f"{fp}/{n_cal} false positives")
    return c, dict(rho=rho, B_obs=B_obs, pB=pB, tv=tv_obs, cap=cap)


def grid_convergence(h):
    lows = []
    for hh in (h, h / 2, h / 4):
        lows.append(np.sort(eigsh(Stadium(hh).hamiltonian(), k=6, sigma=0.0, which="LM")[0]))
    d12, d23 = np.abs(lows[0] - lows[1]), np.abs(lows[1] - lows[2])
    good = int(np.sum(d23 < d12))
    order = np.median(np.log2(d12 / np.maximum(d23, 1e-300)))
    rec("S10", "stadium", "Grid convergence (6 lowest levels)", "ASYMPTOTIC", "Eigenvalues change less on each refinement.",
        float(good), 5.0, good >= 5, note=f"{good}/6 converge; median order {order:.2f}", mode="min")


# =============================================================================
# 8. EXPERIMENTS
# =============================================================================
def phonon_experiments(U, ptab, a):
    s = U.neel
    model = mk(U, s)
    tabs = ptab["tabs"]
    cf = closed_form_gamma(U, s=s)
    om = closed_form_gamma(U, s=+1)["ccw"]
    mg = np.linspace(0, a.mmax_ph, 13)
    wg = np.linspace(min(cf.values()) - 0.25, max(cf.values()) + 0.25, a.nw)
    nW = len(WAVES)
    wi_m, m_m = np.repeat(np.arange(nW), len(mg)), np.tile(mg, nW)
    wi_w, om_w = np.repeat(np.arange(nW), len(wg)), np.tile(wg, nW)
    wi = np.concatenate([wi_m, wi_w])
    omv = np.concatenate([np.full(len(wi_m), om), om_w])
    mv = np.concatenate([m_m, np.full(len(wi_w), a.mspec)])
    res = {}
    for pol in (+1.0, -1.0):
        r = drive_batch(model, pol, omv, mv, wi, tabs)
        res[pol] = r["Win"]
    nm = len(wi_m)
    Ap, Am = res[1.0], res[-1.0]
    C = (Ap - Am) / (Ap + Am)
    return dict(m_grid=mg, w_grid=wg, om=om, branches=cf,
                C_m=C[:nm].reshape(nW, len(mg)), Ap_m=Ap[:nm].reshape(nW, len(mg)), Am_m=Am[:nm].reshape(nW, len(mg)),
                C_w=C[nm:].reshape(nW, len(wg)), Ap_w=Ap[nm:].reshape(nW, len(wg)), Am_w=Am[nm:].reshape(nW, len(wg)))


def scar_experiments(U, lab, c0, stab, a):
    nW = len(WAVES)
    rms = stab[3]
    w0, wgrid, spec = lab.carrier(c0)
    c0v = c0.reshape(-1, 1)
    E_init = lab.energy(c0v)[0]
    disk0 = lab.disk(*U.launch)
    # (i) heating spectroscopy + (ii) memory vs depth, one batch
    wsc = np.linspace(0.5 * w0, 1.5 * w0, a.nws)
    mg = np.linspace(0, a.mmax_sc, 11)
    wi = np.concatenate([np.repeat(np.arange(nW), len(wsc)), np.repeat(np.arange(nW), len(mg)), [0]])
    om = np.concatenate([np.tile(wsc, nW), np.full(nW * len(mg), w0), [w0]])
    mm = np.concatenate([np.full(nW * len(wsc), a.mscar), np.tile(mg, nW), [0.0]])
    eps = np.concatenate([np.full(len(wi) - 1, a.eps), [0.0]])          # last case: undriven, same window
    F = scar_force(stab, wi, om, mm, eps)
    out = lab.evolve(c0, F, a.sdt, disks=[disk0] * len(wi))
    dEs = lab.energy(out["Cf"]) - E_init
    nws = nW * len(wsc)
    heat = dEs[:nws].reshape(nW, len(wsc))
    Bm = out["B"][nws:-1].reshape(nW, len(mg))
    B0 = out["B"][-1]
    # (iii) fractal dimension + maps at (w0, mscar)
    wi2 = np.concatenate([np.arange(nW), [0]])
    F2 = scar_force(stab, wi2, w0, a.mscar, np.concatenate([np.full(nW, a.eps), [0.0]]))
    o2 = lab.evolve(c0, F2, a.sdt, disks=[disk0] * (nW + 1), want_rho=True)
    st = lab.st
    dims, TVs, maps = [], [], []
    for k in range(nW + 1):
        rho = o2["rho"][k]
        TVs.append(0.5 * np.abs(rho - lab.ref).sum())
        X = (rho - lab.ref) / lab.ref.mean()
        thr = X.mean() + X.std()
        grid = st.to_grid(X)
        maps.append(grid)
        dims.append(box_dimension(np.nan_to_num(grid, nan=-9) > thr))
    # (iv) Fibonacci/Vogel atlas, fibword vs chirp
    pts = vogel_launches(U.sigma, a.natlas)
    C0s = np.stack([lab.packet(x, y, U.k0, ang, U.sigma) for x, y, ang in pts], axis=1)
    nP = len(pts)
    wiA = np.concatenate([np.full(nP, WAVES.index("fibword")), np.full(nP, WAVES.index("chirp")), np.full(nP, 0)])
    epsA = np.concatenate([np.full(2 * nP, a.eps), np.zeros(nP)])
    FA = scar_force(stab, wiA, w0, a.mscar, epsA)
    outA = lab.evolve(np.concatenate([C0s] * 3, axis=1), FA, a.sdt,
                      disks=[lab.disk(x, y) for x, y, _ in pts] * 3)
    EA = lab.energy(outA["Cf"]) - lab.energy(np.concatenate([C0s] * 3, axis=1))
    Bfib, Bchirp, Bund = outA["B"][:nP], outA["B"][nP:2 * nP], outA["B"][2 * nP:]
    dfib, dchirp = EA[:nP], EA[nP:2 * nP]
    diff = dfib - dchirp
    obs = diff.mean()
    signs = U.rng_flip.choice([-1.0, 1.0], size=(20000, nP))
    null = signs @ diff / nP
    pval = (1 + np.sum(np.abs(null) >= abs(obs))) / (1 + len(null))
    det = pval < 0.01
    rec("S11", "stadium", "Fibonacci-word FM vs equal-depth chirp: same heating?", "STATISTICAL",
        f"H0: mean over {nP} Vogel-lattice launches of dE(fibword) - dE(chirp) = 0 (paired sign-flip test, alpha=0.01).",
        pval, 0.01, det, status="DETECTED" if det else "NOT DETECTED",
        note=f"mean dE fib {dfib.mean():+.4f}, chirp {dchirp.mean():+.4f}, paired diff {obs:+.4f}, p={pval:.4f}; "
             f"carrier w0={w0:.3f}, depth m={a.mscar}, rms(s): fib {rms[2]:.3f} vs chirp {rms[1]:.3f}")
    return dict(w0=w0, wgrid=wgrid, spec=spec, wsc=wsc, heat=heat, mg=mg, Bm=Bm, B0=B0, dims=dims, TVs=TVs, maps=maps,
                pts=pts, Bfib=Bfib, Bchirp=Bchirp, Bund=Bund, dfib=dfib, dchirp=dchirp, pval=pval, rms=rms, o2=o2)


def scar_volume(U, lab, c0, stab, w0, a):
    nz = a.nz
    mg = np.linspace(0, a.mmax_sc, nz)
    F = scar_force(stab, np.full(nz, WAVES.index("fibword")), w0, mg, a.eps)
    out = lab.evolve(c0, F, a.sdt, want_rho=True)
    st = lab.st
    vol = np.zeros(st.mask.shape + (nz,))
    for k in range(nz):
        vol[:, :, k] = np.nan_to_num(st.to_grid((out["rho"][k] - lab.ref) / lab.ref.mean()), nan=0.0)
    vol = ndimage.zoom(vol, (1, 1, 2), order=1)
    vol = ndimage.gaussian_filter(vol, (1.4, 1.4, 1.2))
    inside = np.repeat(st.mask[:, :, None], vol.shape[2], axis=2)
    lvl = np.percentile(vol[inside], 90)
    return vol, lvl, mg


# =============================================================================
# 9. FINDINGS + CERTIFICATE
# =============================================================================
def findings(P, S, U, a):
    L = []
    cthr = (1.5 - 1) / (1.5 + 1)
    L.append("PHONON: dichroic readout of the hidden Neel order (contrast C = (A+ - A-)/(A+ + A-), readout threshold C = 0.20)")
    for i, w in enumerate(WAVES):
        c = np.abs(P["C_m"][i])
        below = np.where(c < cthr)[0]
        first = f"first drops below threshold at m={P['m_grid'][below[0]]:.3f}" if len(below) else "stays above threshold over the whole sweep"
        decoded = np.sign(P["C_m"][i]) == np.sign(P["C_m"][0])
        L.append(f"  {WNAME[w]:26s} C(m=0)={P['C_m'][i][0]:+.3f}  C(m_max)={P['C_m'][i][-1]:+.3f}  {first}; "
                 f"sign preserved at {int(decoded.sum())}/{len(decoded)} depths")
    L.append("PHONON: dichroism spectrum (carrier sweep at fixed small depth)")
    wb = P["branches"]
    for i, w in enumerate(WAVES):
        Cw = P["C_w"][i]
        L.append(f"  {WNAME[w]:26s} max C={Cw.max():+.3f} at w={P['w_grid'][np.argmax(Cw)]:.3f}, min C={Cw.min():+.3f} at w={P['w_grid'][np.argmin(Cw)]:.3f}"
                 f"   (branches: CCW {wb['ccw']:.3f}, CW {wb['cw']:.3f})")
    L.append("SCAR: heating, memory and fractal structure at the dipole-resonant carrier")
    cw_heat = np.trapezoid(np.abs(S["heat"][0]), S["wsc"])
    for i, w in enumerate(WAVES):
        hint = np.trapezoid(np.abs(S["heat"][i]), S["wsc"])
        L.append(f"  {WNAME[w]:26s} integrated |dE| over carrier sweep = {hint / (cw_heat + 1e-30):.2f} x CW | "
                 f"B at depth m={a.mscar}: {S['Bm'][i][np.argmin(np.abs(S['mg'] - a.mscar))]:.2f} (undriven same window {S['B0']:.2f}) | "
                 f"TV {S['TVs'][i]:.3f} | ridge box-dimension {S['dims'][i]:.2f}")
    L.append(f"  undriven reference               TV {S['TVs'][-1]:.3f} | ridge box-dimension {S['dims'][-1]:.2f}")
    L.append(f"SCAR ATLAS on {len(S['pts'])} golden-angle launch points: Fibonacci-word vs chirp heating -> "
             f"{'DETECTED difference' if S['pval'] < 0.01 else 'no significant difference'} (p={S['pval']:.4f}); "
             f"mean B under fibword {S['Bfib'].mean():.2f}, chirp {S['Bchirp'].mean():.2f}, undriven {S['Bund'].mean():.2f}")
    L.append("NOTE: depth m is the PEAK fractional detuning; waveforms differ in rms detuning, so they differ in effective bandwidth.")
    L.append("      rms(s) = " + ", ".join(f"{w} {r:.2f}" for w, r in zip(WAVES, S["rms"])))
    return L


def build_certificate(U, args, runtime, extra, flines):
    counted = [r for r in RESULTS if r["status"] in ("PASS", "FAIL")]
    fails = [r["id"] for r in counted if r["status"] == "FAIL"]
    core = {"text_sha256": U.digest, "params": U.params(), "verdicts": {r["id"]: r["status"] for r in RESULTS}}
    digest = hashlib.sha256(json.dumps(core, sort_keys=True, default=float).encode()).hexdigest()
    me = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
    return dict(program="qnos_fibonacci_lab.py", version=VERSION, program_sha256=me, text=U.text, text_sha256=U.digest,
                params=U.params(), environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                                                   matplotlib=matplotlib.__version__, marching_cubes="skimage" if HAVE_SK else "builtin voxel mesher",
                                                   platform=platform.platform()),
                settings=vars(args), runtime_s=round(runtime, 1), results=RESULTS,
                passed=sum(r["status"] == "PASS" for r in RESULTS), failed=fails,
                not_applicable=[r["id"] for r in RESULTS if r["status"] == "N/A"],
                statistical={r["id"]: r["status"] for r in RESULTS if r["kind"] == "STATISTICAL"},
                findings=flines, extra=extra, certificate_digest=digest)


def write_text_report(cert, path):
    L = ["QNOS FIBONACCI LAB -- VERIFICATION CERTIFICATE", "=" * 78,
         f"text            : {cert['text']!r}", f"sha256(text)    : {cert['text_sha256']}",
         f"program sha256  : {cert['program_sha256'][:16]}...",
         f"environment     : python {cert['environment']['python']}, numpy {cert['environment']['numpy']}, scipy {cert['environment']['scipy']}, mesher {cert['environment']['marching_cubes']}",
         f"runtime         : {cert['runtime_s']} s", "", "Configuration selected by the text"]
    for k, v in cert["params"].items():
        L.append(f"  {k:8s} = {v if not isinstance(v, float) else round(v, 6)}")
    L.append("")
    for grp in ("phonon", "modulation", "stadium"):
        L.append(f"[{grp.upper()}]")
        for r in (x for x in cert["results"] if x["group"] == grp):
            rel = "<=" if r["mode"] == "max" else ">="
            L.append(f"  {r['id']:3s} {r['status']:12s} {r['kind']:11s} {r['title']}")
            L.append(f"        claim   : {r['claim']}")
            L.append(f"        measured: {r['measured']:.4g}  (criterion {rel} {r['tol']:.4g})")
            if r["note"]:
                L.append(f"        detail  : {r['note']}")
        L.append("")
    L += ["SUMMARY", f"  passed {cert['passed']}, failed {cert['failed'] or 'none'}, not applicable {cert['not_applicable'] or 'none'}",
          f"  statistical verdicts: {cert['statistical']}", f"  certificate digest: {cert['certificate_digest']}", "",
          "FINDINGS (measurements of the model, generated from the numbers above)"] + ["  " + x if not x.startswith(" ") else x for x in cert["findings"]]
    L += ["", "INTERPRETATION LIMITS",
          "  * Numerical verification at one text-selected parameter point, not formal proof.",
          "  * Minimal lattice-dynamics model and a projected dipole drive; not an ab-initio material calculation.",
          "  * The scar drive acts inside the computed eigenbasis window (captured weight reported in S3); results describe that projection.",
          "  * 'DETECTED' / 'NOT DETECTED' are hypothesis-test outcomes; both are valid results.",
          "  * Run several texts: IDENTITY / CLOSED-FORM / BOUND claims must pass for every one."]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return "\n".join(L)


# =============================================================================
# 10. DASHBOARD
# =============================================================================
def dashboard(U, cert, P, S, SPEC, VOLS, path):
    fig = plt.figure(figsize=(25, 15.5))
    gs = GridSpec(3, 4, figure=fig, hspace=0.34, wspace=0.34, left=0.04, right=0.985, top=0.93, bottom=0.02, height_ratios=[1, 1, 1.45])
    wb = P["branches"]
    sub = gs[0, 0:2].subgridspec(2, 3, hspace=0.38, wspace=0.12)
    for i, w in enumerate(WAVES):
        ax = fig.add_subplot(sub[i // 3, i % 3])
        f, tt, Sxx = SPEC[w]
        ax.pcolormesh(tt, f, 10 * np.log10(Sxx + 1e-30), shading="gouraud", cmap="magma",
                      vmin=10 * np.log10(Sxx.max() + 1e-30) - 45, vmax=10 * np.log10(Sxx.max() + 1e-30))
        for v_ in wb.values():
            ax.axhline(v_, color="#00f0ff", lw=0.6, ls="--")
        ax.set_ylim(min(wb.values()) - 0.55, max(wb.values()) + 0.55)
        ax.set_title(WNAME[w], color=COL[w], fontsize=8)
        if i % 3 == 0: ax.set_ylabel("w")
        if i // 3 == 1: ax.set_xlabel("t")
    fig.text(0.04, 0.951, "Drive spectrograms: time (x) vs frequency (y); dashed = CCW / CW Gamma branches", color="#7fd", fontsize=8)

    ax = fig.add_subplot(gs[0, 2])
    for i, w in enumerate(WAVES):
        ax.plot(P["m_grid"], np.abs(P["C_m"][i]), "o-", ms=3, color=COL[w], label=WNAME[w], lw=1.4)
    ax.axhline(0.2, color="w", ls=":", lw=0.8)
    ax.text(P["m_grid"][-1], 0.21, "readout threshold (ratio 1.5)", ha="right", fontsize=6.5, color="w")
    ax.set_xlabel("peak fractional detuning m"); ax.set_ylabel("|dichroic contrast|")
    ax.set_title("Phonon readout of hidden order vs modulation depth"); ax.legend(loc="lower left")

    ax = fig.add_subplot(gs[0, 3])
    for i, w in enumerate(WAVES):
        ax.plot(P["w_grid"], P["C_w"][i], color=COL[w], lw=1.4, label=WNAME[w])
    for k_, v_ in wb.items():
        ax.axvline(v_, color="#00f0ff", ls="--", lw=0.7)
    ax.axhline(0, color="#456", lw=0.6)
    ax.set_xlabel("carrier w0"); ax.set_ylabel("C = (A+ - A-)/(A+ + A-)")
    ax.set_title(f"Dichroism spectrum (m={cert['settings']['mspec']}): sign flips between branches")

    ax = fig.add_subplot(gs[1, 0])
    for i, w in enumerate(WAVES):
        ax.plot(S["wsc"] / S["w0"], S["heat"][i], color=COL[w], lw=1.3, label=WNAME[w])
    ax.set_xlabel("carrier / dipole-resonant w0"); ax.set_ylabel("absorbed energy dE")
    ax.set_title("Scar heating spectroscopy (stadium, dipole drive)")

    ax = fig.add_subplot(gs[1, 1])
    for i, w in enumerate(WAVES):
        ax.plot(S["mg"], S["Bm"][i] / S["B0"], "o-", ms=3, color=COL[w], lw=1.3)
    ax.axhline(1, color="w", ls=":", lw=0.7)
    ax.set_xlabel("modulation depth m"); ax.set_ylabel("birthmark B / B(undriven)")
    ax.set_title("Memory of the launch under modulation")

    ax = fig.add_subplot(gs[1, 2])
    st = SPEC["_stadium"]
    ax.contour(st.X, st.Y, st.mask.astype(float), levels=[0.5], colors="#35505a", linewidths=1)
    px, py = np.array([p[0] for p in S["pts"]]), np.array([p[1] for p in S["pts"]])
    ratio = S["Bfib"] / S["Bund"]
    sc = ax.scatter(px, py, c=ratio, s=60 + 140 * (S["dfib"] - S["dchirp"] - (S["dfib"] - S["dchirp"]).min()) /
                    (np.ptp(S["dfib"] - S["dchirp"]) + 1e-30), cmap="plasma", edgecolors="w", linewidths=0.4)
    plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, label="B(fibword)/B(undriven)")
    ax.set_aspect("equal"); ax.set_title(f"Birthmark atlas, golden-angle launches (p={S['pval']:.3f})")

    ax = fig.add_subplot(gs[1, 3])
    names = [WNAME[w].split(" (")[0] for w in WAVES] + ["undriven"]
    cols = [COL[w] for w in WAVES] + ["#ffffff"]
    ax.barh(range(len(names)), S["dims"], color=cols)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7); ax.invert_yaxis()
    ax.set_xlim(1.0, 2.0); ax.set_xlabel("box-counting dimension of birthmark ridges")
    ax.set_title("Fractal dimension of the scar / birthmark ridge set")

    (vp, vm_), (verts_p, faces_p, verts_m, faces_m), (sv, sf) = VOLS["phonon"], VOLS["phonon_mesh"], VOLS["scar_mesh"]
    ax = fig.add_subplot(gs[2, 0:2], projection="3d")
    res, dep = vp.shape[0], vp.shape[2]
    style3d(ax, "Amduat-MRI worldline tubes (x=u_x, y=u_y, z=time): sigma+ (left) vs sigma- (right), Fibonacci-FM pump",
            ((0, 2.5 * res), (0, res), (0, dep)))
    draw_mesh(ax, verts_p, faces_p, cm.cool, 2, maxf=16000)
    draw_mesh(ax, verts_m, faces_m, cm.autumn, 2, shift=(1.4 * res, 0, 0), maxf=16000)
    ax.view_init(20, -66)
    ax = fig.add_subplot(gs[2, 2:4], projection="3d")
    ext = VOLS["scar_extent"]
    style3d(ax, "Scar volume: x,y = stadium, z = Fibonacci-FM depth m (isosurface of excess time-averaged density)",
            ((0, ext[0]), (0, ext[1]), (0, ext[2])))
    draw_mesh(ax, sv, sf, cm.inferno, 2, maxf=16000)
    ax.view_init(24, -58)
    st_txt = "  ".join(f"{k}:{v}" for k, v in cert["statistical"].items())
    fig.suptitle(f"QNOS FIBONACCI LAB | {U.text[:48]!r} | {cert['passed']} claims passed, failed: {cert['failed'] or 'none'} | "
                 f"hidden order {U.neel:+d} | {st_txt} | digest {cert['certificate_digest'][:16]}", color="#fff", fontsize=12)
    fig.savefig(path, dpi=110)
    plt.close(fig)


# =============================================================================
# 11. DRIVER
# =============================================================================
def run(text, a):
    RESULTS.clear()
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    U = SeededUniverse(text)
    tag = U.digest[:10]
    print(f"text sha256 {U.digest[:16]}...  mB/mA={U.mB:.3f} g={U.g:.3f} Neel {U.neel:+d} k0={U.k0:.2f} launch=({U.launch[0]:.2f},{U.launch[1]:.2f})")
    # tables
    t, dt, n = phonon_grid()
    J, E, rms_p = make_tables(t, 160.0, seed=U.wave_seed)
    ptab = dict(grid=(t, dt, n), tabs=(t, J, E))
    print("modulation claims ..."); modulation_claims()
    print("phonon claims ..."); om = phonon_claims(U, ptab, a)
    print("phonon sweeps (6 modulations x depth x carrier, both helicities) ...")
    P = phonon_experiments(U, ptab, a)
    # spectrograms of the drives (Ex component)
    SPEC = {}
    tf = np.arange(0, 1.35 * 160, 0.05)
    Jf, Ef, _ = make_tables(tf, 160.0, seed=U.wave_seed)
    for i, w in enumerate(WAVES):
        x = Ef[:, i] * np.cos(om * (tf + a.mspec_show * Jf[:, i]))
        f, tt, Sxx = signal.spectrogram(x, fs=1 / 0.05, nperseg=2048, noverlap=1900)
        SPEC[w] = (2 * np.pi * f, tt, Sxx)
    # phonon volumes
    print("phonon volumes ...")
    mdl = mk(U, U.neel)
    vols, curves = phonon_volumes(mdl, om, a.mspec_show, WAVES.index("fibword"), ptab["tabs"])
    vp, vm_ = vols
    vertp, facep = mesh_from_volume(vp, 0.22)
    vertm, facem = mesh_from_volume(vm_, 0.22)
    write_obj(os.path.join(a.out, f"phonon_sigma_plus_{tag}.obj"), vertp, facep, "QNOS Fibonacci Lab phonon worldline tube, sigma+")
    write_obj(os.path.join(a.out, f"phonon_sigma_minus_{tag}.obj"), vertm, facem, "QNOS Fibonacci Lab phonon worldline tube, sigma-")
    # scar
    print(f"stadium eigenstates (h={a.h}, {a.nev} states) ...")
    lab = ScarLab(a.h, a.nev, U.k0)
    c0 = lab.packet(*U.launch, U.k0, U.angle, U.sigma)
    stab = scar_tables(a.sT, a.sdt, U.wave_seed)
    w0 = lab.carrier(c0)[0]
    print(f"dipole-resonant carrier w0 = {w0:.3f}")
    print("stadium claims ...")
    dts = [a.sdt * 4, a.sdt * 2, a.sdt]
    c, birth = scar_claims(U, lab, a, stab, w0, dts)
    if not a.quick:
        print("grid-convergence claim ..."); grid_convergence(a.h)
    print("scar experiments (heating spectroscopy, memory, fractal dimension, Vogel atlas) ...")
    S = scar_experiments(U, lab, c0, stab, a)
    print("scar volume ...")
    sv_vol, lvl, mg = scar_volume(U, lab, c0, stab, S["w0"], a)
    sv, sf = mesh_from_volume(sv_vol, lvl, spacing=(1.0, 1.0, 1.0))
    # swap (y,x,z) -> (x,y,z) for display
    sv = sv[:, [1, 0, 2]] if len(sv) else sv
    zs = 0.45 * sv_vol.shape[1] / sv_vol.shape[2]                          # display height = 0.45 x width
    write_obj(os.path.join(a.out, f"scar_volume_{tag}.obj"), sv * np.array([lab.st.h, lab.st.h, lab.st.h * zs]), sf,
              "QNOS Fibonacci Lab scar volume (z = modulation depth, display-scaled)")
    sv = sv * np.array([1.0, 1.0, zs])
    flines = findings(P, S, U, a)
    extra = dict(w0_scar=S["w0"], phonon_pump=om, rms_phonon=dict(zip(WAVES, map(float, rms_p))), fractal_dims=dict(zip(WAVES + ["undriven"], map(float, S["dims"]))),
                 mesh=dict(phonon_plus=[len(vertp), len(facep)], phonon_minus=[len(vertm), len(facem)], scar=[len(sv), len(sf)]))
    cert = build_certificate(U, a, time.time() - t0, extra, flines)
    with open(os.path.join(a.out, f"fib_certificate_{tag}.json"), "w") as fh:
        json.dump(cert, fh, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    np.savez_compressed(os.path.join(a.out, f"fib_data_{tag}.npz"), **{f"phonon_{k}": v for k, v in P.items() if isinstance(v, np.ndarray)},
                        **{f"scar_{k}": np.asarray(v) for k, v in S.items() if k in ("wsc", "heat", "mg", "Bm", "Bfib", "Bchirp", "Bund", "dfib", "dchirp", "dims", "TVs", "wgrid", "spec")},
                        scar_volume=sv_vol)
    txt = write_text_report(cert, os.path.join(a.out, f"fib_certificate_{tag}.txt"))
    print(txt)
    SPEC["_stadium"] = lab.st
    ext = (sv_vol.shape[1], sv_vol.shape[0], 0.45 * sv_vol.shape[1])
    VOLS = dict(phonon=(vp, vm_), phonon_mesh=(vertp, facep, vertm, facem), scar_mesh=(sv, sf), scar_extent=ext)
    fp = os.path.join(a.out, f"fib_dashboard_{tag}.png")
    dashboard(U, cert, P, S, SPEC, VOLS, fp)
    print(f"\nwrote {fp}\n      fib_certificate_{tag}.txt/.json, fib_data_{tag}.npz, phonon_sigma_plus/minus_{tag}.obj, scar_volume_{tag}.obj in {a.out}/")
    if "--show" in sys.argv:
        plt.show()
    return cert


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="?")
    ap.add_argument("--out", default="qnos_fib_out")
    ap.add_argument("--h", type=float, default=0.03, help="billiard grid spacing")
    ap.add_argument("--nev", type=int, default=220, help="billiard eigenstates")
    ap.add_argument("--shuffles", type=int, default=2000)
    ap.add_argument("--eps", type=float, default=4.0, help="scar dipole drive strength")
    ap.add_argument("--mscar", type=float, default=0.15, help="scar modulation depth for single-point runs")
    ap.add_argument("--mmax-sc", dest="mmax_sc", type=float, default=0.5)
    ap.add_argument("--mmax-ph", dest="mmax_ph", type=float, default=0.25)
    ap.add_argument("--mspec", type=float, default=0.03, help="phonon depth for the dichroism spectrum")
    ap.add_argument("--mspec-show", dest="mspec_show", type=float, default=0.06, help="phonon depth for spectrograms and volumes")
    ap.add_argument("--nw", type=int, default=48, help="phonon carrier-sweep points")
    ap.add_argument("--nws", type=int, default=48, help="scar carrier-sweep points")
    ap.add_argument("--natlas", type=int, default=55, help="Vogel-lattice launch points (Fibonacci number suggested)")
    ap.add_argument("--nz", type=int, default=24, help="modulation-depth slices in the scar volume")
    ap.add_argument("--sT", type=float, default=12.0, help="scar drive duration")
    ap.add_argument("--sdt", type=float, default=0.004, help="scar time step")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.h, a.nev, a.shuffles, a.nw, a.nws, a.natlas, a.nz, a.sT, a.sdt = 0.04, 120, 500, 24, 24, 34, 14, 8.0, 0.005
    text = a.text or input("> Enter text to seed the physics: ").strip()
    if not text:
        sys.exit("Text required.")
    cert = run(text, a)
    sys.exit(1 if cert["failed"] else 0)


if __name__ == "__main__":
    main()
