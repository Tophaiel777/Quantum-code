#!/usr/bin/env python3
"""
QNOS PROOFS -- text-seeded physics with a verification certificate
===================================================================
Requires: numpy, scipy, matplotlib  and  qnos_physics20262.py (or qnos_physics.py) in the same folder.

    python qnos_proofs.py "any text you like"
    python qnos_proofs.py "any text" --out results --quick
    python qnos_proofs.py                       # interactive prompt, like ImplicitGenerator

WHAT THE TEXT DOES
------------------
The text is hashed (SHA-256) and the hash seeds a NumPy generator. That generator fixes ONE physical
configuration of the QNOS models: sublattice masses, spring constants, spin-phonon coupling, Neel sign
(the hidden order), and the stadium wavepacket (launch point, direction, wavenumber, width).
Unlike ImplicitGenerator, the text never decides what the output "looks like". It only chooses the point in
parameter space at which the physics is tested.

WHAT A "PROOF" MEANS HERE (read this)
-------------------------------------
This program does computational verification, not formal mathematical proof. Every result is a claim
that is derived (or stated by a theorem/known law) and then tested against the exact numerics, with a
pre-declared tolerance, for the configuration the text selected. Each claim carries a kind:

  IDENTITY     exact mathematical identity; must hold to rounding error (tol ~1e-9 or tighter).
  CLOSED-FORM  numerics compared with an analytic result derived in the comments below.
  BOUND        a rigorous inequality; the measured quantity must lie below the bound.
  ASYMPTOTIC   a law valid only in a limit (Weyl law); tolerance reflects finite-size error.
  DYNAMICS     a property of the time-integrated ODE; tolerance reflects integrator error.
  STATISTICAL  a hypothesis test with a stated null; the verdict may legitimately be "NOT DETECTED".
  CONTROL      a deliberately broken system: the test MUST fail, which shows the tests have power.

A PASS on one text is evidence for that parameter point only. Run many texts: every IDENTITY/CLOSED-FORM
/BOUND claim must pass for all of them, and a single failure falsifies the claim or exposes a bug.
The certificate stores the hashes and parameters, so any result can be reproduced exactly.

DERIVATIONS USED
----------------
(D1) Gamma-point closed form. At k=0 the honeycomb force matrix is Phi0 = [[S,-S],[-S,S]] with
     S = kappa*I, kappa = 3/2 (phi_l+phi_t) (three bond directions 120 deg apart). In circular
     coordinates u_pm = u_x +- i u_y the gyroscopic matrix J acts as -+i, and with u ~ exp(-i w t):
         (mA w^2 -/+ g s w - kappa) a + kappa b = 0,  (mB w^2 +/- g s w - kappa) b + kappa a = 0.
     Eliminating a, b gives  w^2 [ mA mB w^2 + g s (mA-mB) w - (kappa (mA+mB) + g^2) ] = 0  (CW),
     and the same with s -> -s (CCW). Hence
         w_CCW,CW = ( +/- g s (mA-mB) + D ) / (2 mA mB),  D = sqrt( g^2 (mA-mB)^2 + 4 mA mB (kappa (mA+mB) + g^2) )
         w_CCW - w_CW = g s (1/mB - 1/mA)      (exactly linear in g, odd in s, zero for mA = mB).
(D2) Time reversal. Complex conjugation of (Phi(k) - w^2 M - i w G) e = 0 gives the same problem at -k
     with G -> -G (i.e. Neel sign flipped), so  w_n(k, s) = w_n(-k, -s).
(D3) Trace sum rule. For A = [[0, I],[-M^-1 Phi, M^-1 G]],  tr(A^2) = -2 tr(M^-1 Phi) + tr((M^-1 G)^2), and the
     8 eigenvalues at k split into the 4 positive frequencies at k and at -k, so
         sum_j [ w_j^2(k) + w_j^2(-k) ] = 2 tr(S) (1/mA + 1/mB) + 2 g^2 (1/mA^2 + 1/mB^2).
(D4) Mirror x -> -x maps the lattice to itself, flips G (so s -> -s) and maps sigma+ light to -(sigma- light).
     Therefore |u(t)| for (pump, s) equals |u(t)| for (-pump, -s).
(D5) Time-average bound. With c_n the overlaps and psi_n real, normalised eigenstates, the exact time average
     over [0,T] differs from the diagonal ensemble rho_bar = sum |c_n|^2 psi_n^2 by
         || <rho>_T - rho_bar ||_1 <= (2/T) sum_{n!=m} |c_n c_m| / |E_n - E_m|,
     since |(1/T) int_0^T e^{-i dE t} dt| <= 2/(T|dE|) and ||psi_n psi_m||_1 <= 1 (Cauchy-Schwarz).
"""
import argparse, hashlib, importlib.util, json, math, os, platform, sys, time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy
from scipy.sparse.linalg import eigsh

VERSION = "1.0"


# =============================================================================
# 0. LOAD THE ORIGINAL QNOS PHYSICS MODULE
# =============================================================================
def load_physics(explicit=None):
    here = os.path.dirname(os.path.abspath(__file__))
    cands = ([explicit] if explicit else []) + [os.path.join(here, n) for n in
                                                ("qnos_physics20262.py", "qnos_physics.py")]
    for p in cands:
        if p and os.path.isfile(p):
            spec = importlib.util.spec_from_file_location("qnos_physics_mod", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            with open(p, "rb") as f:
                return mod, p, hashlib.sha256(f.read()).hexdigest()
    sys.exit("Could not find qnos_physics20262.py / qnos_physics.py next to this script (or use --physics PATH).")


# =============================================================================
# 1. TEXT -> PHYSICAL CONFIGURATION
# =============================================================================
class SeededUniverse:
    """Deterministic map  text -> physical configuration. Same text, same universe, on every machine."""

    def __init__(self, text, R=1.0, Ls=1.0):
        self.text = text
        self.digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        self.seed = int(self.digest, 16)
        r = np.random.default_rng(np.random.SeedSequence(self.seed))
        U = lambda a, b: float(r.uniform(a, b))
        # --- phonon lattice (ranges keep the Gamma splitting resolvable against the pulse bandwidth)
        self.mA = 1.0
        self.mB = U(1.6, 3.0)
        self.phi_l = U(0.8, 1.4)
        self.phi_t = self.phi_l * U(0.25, 0.5)
        self.g = U(0.25, 0.45)
        self.neel = int(r.choice([-1, 1]))            # the hidden order the optical readout must recover
        self.kappa = 1.5 * (self.phi_l + self.phi_t)
        # --- stadium wavepacket
        self.k0, self.sigma, self.angle = U(15.0, 20.0), U(0.14, 0.22), U(0.0, 360.0)
        for _ in range(100000):
            x, y = U(-1.7, 1.7), U(-0.7, 0.7)
            if R - math.hypot(x - min(max(x, -Ls), Ls), y) > 3 * self.sigma:
                break
        self.launch = (x, y)
        # independent streams so adding a test never changes another test's random numbers
        self.rng_k = np.random.default_rng(np.random.SeedSequence([self.seed & (2**128 - 1), 1]))
        self.rng_perm = np.random.default_rng(np.random.SeedSequence([self.seed & (2**128 - 1), 2]))

    def params(self):
        keys = ("mA", "mB", "phi_l", "phi_t", "g", "neel", "k0", "sigma", "angle", "launch")
        return {k: getattr(self, k) for k in keys}


# =============================================================================
# 2. RESULT RECORDS
# =============================================================================
RESULTS = []


def rec(rid, group, title, kind, claim, measured, tol, passed, note="", status=None, mode="max"):
    """mode 'max': passes when measured <= tol.  mode 'min': passes when measured >= tol."""
    r = dict(id=rid, group=group, title=title, kind=kind, claim=claim, measured=float(measured), tol=float(tol),
             mode=mode, status=status or ("PASS" if passed else "FAIL"), note=note)
    RESULTS.append(r)
    return r


def closed_form_gamma(U, g=None, mB=None, s=None):
    g = U.g if g is None else g
    mA, mB = U.mA, (U.mB if mB is None else mB)
    s = U.neel if s is None else s
    D = math.sqrt(g * g * (mA - mB) ** 2 + 4 * mA * mB * (U.kappa * (mA + mB) + g * g))
    return dict(ccw=(g * s * (mA - mB) + D) / (2 * mA * mB), cw=(-g * s * (mA - mB) + D) / (2 * mA * mB))


def mk(Q, U, s, g=None, mB=None):
    return Q.PhononModel(mA=U.mA, mB=U.mB if mB is None else mB, phi_l=U.phi_l, phi_t=U.phi_t,
                         g=U.g if g is None else g, neel=s)


# =============================================================================
# 3. PHONON CLAIMS
# =============================================================================
def phonon_static_claims(Q, U, nk=24):
    u, v = U.rng_k.random((2, nk))
    ks = u[:, None] * Q.PhononModel.b1 + v[:, None] * Q.PhononModel.b2
    mp, mm = mk(Q, U, +1), mk(Q, U, -1)
    s = U.neel
    ms = mp if s > 0 else mm

    # P1 algebraic structure
    herm = max(np.abs(ms.phi(k) - ms.phi(k).conj().T).max() for k in ks)
    real = max(np.abs(ms.phi(-k) - ms.phi(k).conj()).max() for k in ks)
    psd = max(0.0, -min(np.linalg.eigvalsh(ms.phi(k)).min() for k in ks))
    anti = np.abs(ms.G + ms.G.T).max()
    rec("P1", "phonon", "Force-matrix structure", "IDENTITY",
        "Phi(k) Hermitian, Phi(-k)=conj Phi(k), Phi >= 0 (translation invariance), G real antisymmetric.",
        max(herm, real, psd, anti), 1e-12, max(herm, real, psd, anti) <= 1e-12)

    # P2 dynamical stability: every eigenvalue of the 8x8 linearisation is purely imaginary
    Z, I4 = np.zeros((4, 4)), np.eye(4)
    worst = 0.0
    for k in ks:
        A = np.block([[Z, I4], [-ms.Minv @ ms.phi(k), ms.Minv @ ms.G]])
        worst = max(worst, np.abs(np.linalg.eigvals(A).real).max())
    rec("P2", "phonon", "Dynamical stability", "IDENTITY",
        "Hermitian PSD Phi, M>0, antisymmetric G  =>  all w real (max |Re lambda| = 0).", worst, 1e-8, worst <= 1e-8)

    # P3 time reversal  w(k,s) = w(-k,-s)
    tr = max(np.abs(mp.solve(k)[0] - mm.solve(-k)[0]).max() for k in ks)
    rec("P3", "phonon", "Time-reversal relation (D2)", "IDENTITY",
        "w_n(k, s) = w_n(-k, -s) for all k, n.", tr, 1e-9, tr <= 1e-9)

    # P4 time reversal is actually broken in a fixed Neel state, and restored at g=0
    nr = max(np.abs(ms.solve(k)[0] - ms.solve(-k)[0]).max() for k in ks)
    m0 = mk(Q, U, s, g=0.0)
    nr0 = max(np.abs(m0.solve(k)[0] - m0.solve(-k)[0]).max() for k in ks)
    rec("P4", "phonon", "Non-reciprocity (T-breaking) and its g=0 null", "CONTROL",
        "max_k |w(k)-w(-k)| > 0 at g>0 (effect) while exactly 0 at g=0 (null).", nr, 1e-3,
        nr > 1e-3 and nr0 <= 1e-10, note=f"effect={nr:.4g}, null at g=0 = {nr0:.2e}", mode="min")

    # P5 trace sum rule (D3)
    rhs = 2 * np.trace(ms.S) * (1 / U.mA + 1 / U.mB) + 2 * U.g ** 2 * (1 / U.mA ** 2 + 1 / U.mB ** 2)
    sr = max(abs((ms.solve(k)[0] ** 2).sum() + (ms.solve(-k)[0] ** 2).sum() - rhs) / rhs for k in ks)
    rec("P5", "phonon", "Trace sum rule (D3)", "IDENTITY",
        "sum_j[w_j^2(k)+w_j^2(-k)] = 2 tr(S)(1/mA+1/mB) + 2 g^2 (1/mA^2+1/mB^2), independent of k.",
        sr, 1e-10, sr <= 1e-10, note=f"RHS={rhs:.6f}")

    # G1..G3 Gamma-point closed form
    def num_gamma(m):
        w, _, l = m.solve(np.zeros(2))
        o = [2, 3]
        up = [j for j in o if l[j] > 0]
        dn = [j for j in o if l[j] < 0]
        if len(up) != 1 or len(dn) != 1:
            return None
        return dict(ccw=w[up[0]], cw=w[dn[0]], l=(l[up[0]], l[dn[0]]))
    ng, cf = num_gamma(ms), closed_form_gamma(U)
    if ng is None:
        rec("G1", "phonon", "Gamma optical doublet (D1)", "CLOSED-FORM", "one CCW and one CW optical mode", np.inf, 1e-9, False)
    else:
        err = max(abs(ng["ccw"] - cf["ccw"]), abs(ng["cw"] - cf["cw"]), abs(abs(ng["l"][0]) - 1), abs(abs(ng["l"][1]) - 1))
        flipped = closed_form_gamma(U, s=-s)
        sep = max(abs(ng["ccw"] - flipped["ccw"]), abs(ng["cw"] - flipped["cw"]))
        rec("G1", "phonon", "Gamma optical doublet vs analytic formula (D1)", "CLOSED-FORM",
            "w_CCW,CW = (+/- g s (mA-mB) + D)/(2 mA mB); modes are purely circular (|ell| = 1).",
            err, 1e-9, err <= 1e-9 and sep > 1e-3,
            note=f"numeric CCW={ng['ccw']:.8f} CW={ng['cw']:.8f}; closed form CCW={cf['ccw']:.8f} CW={cf['cw']:.8f}; "
                 f"wrong-sign formula would miss by {sep:.3g}")
    ng_o = num_gamma(mm if s > 0 else mp)
    if ng is not None and ng_o is not None:
        d1, d2 = ng["ccw"] - ng["cw"], ng_o["ccw"] - ng_o["cw"]
        exact = U.g * s * (1 / U.mB - 1 / U.mA)
        e = max(abs(d1 + d2), abs(d1 - exact))
        rec("G2", "phonon", "Splitting is odd in Neel order and linear in g (D1)", "CLOSED-FORM",
            "w_CCW - w_CW = g s (1/mB - 1/mA); reverses sign with the Neel order.", e, 1e-9, e <= 1e-9,
            note=f"splitting(s)={d1:+.8f}, splitting(-s)={d2:+.8f}, formula={exact:+.8f}")
    me = mk(Q, U, s, mB=U.mA)
    wp = np.abs(me.solve(np.zeros(2))[0][2] - me.solve(np.zeros(2))[0][3])
    rec("G3", "phonon", "Inversion asymmetry is required (mA = mB null)", "CLOSED-FORM",
        "With equal sublattice masses the Gamma doublet stays degenerate for any g.", wp, 1e-8, wp <= 1e-8,
        note=f"|w3-w2| = {wp:.2e} with mB=mA, g={U.g:.3f}")
    return ks


def phonon_drive_claims(Q, U):
    s = U.neel
    Om = closed_form_gamma(U, s=+1)["ccw"]          # pump calibrated once, on the reference (s=+1) CCW branch
    amp, store = {}, {}
    for pol in (+1, -1):
        for sg in (+1, -1):
            t, u, _ = mk(Q, U, sg).drive(pol, Om)
            amp[(pol, sg)] = np.maximum(np.hypot(u[:, 0], u[:, 1]), np.hypot(u[:, 2], u[:, 3]))
            store[(pol, sg)] = (t, u)
    sc = max(a.max() for a in amp.values())
    mirror = max(np.abs(amp[(+1, +1)] - amp[(-1, -1)]).max(), np.abs(amp[(+1, -1)] - amp[(-1, +1)]).max()) / sc
    rec("D1", "phonon", "Mirror covariance of the driven response (D4)", "DYNAMICS",
        "|u(t)| for (pump, s) equals |u(t)| for (-pump, -s).", mirror, 1e-6, mirror <= 1e-6,
        note=f"pump frequency {Om:.6f} (analytic)")

    # linearity
    t, u1, _ = mk(Q, U, s).drive(+1, Om, E0=0.02)
    _, u2, _ = mk(Q, U, s).drive(+1, Om, E0=0.04)
    lin = np.abs(u2 - 2 * u1).max() / np.abs(u2).max()
    rec("D2", "phonon", "Linear response", "DYNAMICS", "Doubling the field doubles the displacement.",
        lin, 1e-6, lin <= 1e-6)

    # hidden-order readout: decide the sign of s from the circular dichroism alone
    pk = {k: a.max() for k, a in amp.items()}
    delta = abs(U.g * (1 / U.mB - 1 / U.mA))
    res_ok, detail = True, []
    for sg in (+1, -1):
        decoded = +1 if pk[(+1, sg)] > pk[(-1, sg)] else -1
        ratio = max(pk[(+1, sg)], pk[(-1, sg)]) / min(pk[(+1, sg)], pk[(-1, sg)])
        res_ok &= (decoded == sg and ratio >= 1.5)
        detail.append(f"s={sg:+d}: decoded {decoded:+d}, dichroic ratio {ratio:.2f}")
    if delta < 0.08:
        rec("D3", "phonon", "Hidden-order readout from circular dichroism", "DYNAMICS",
            "Matched helicity responds more strongly, revealing the sign of the Neel order.", delta, 0.08, False,
            status="N/A", note=f"Gamma splitting {delta:.3f} not resolvable against pulse bandwidth; " + "; ".join(detail), mode="min")
    else:
        rec("D3", "phonon", "Hidden-order readout from circular dichroism", "DYNAMICS",
            "Matched helicity responds more strongly, revealing the sign of the Neel order (needs ratio >= 1.5).",
            float(res_ok), 1.0, res_ok, note="; ".join(detail) + f"; seeded hidden order was {s:+d}", mode="min")

    # control: break the x-mirror by stiffening one bond; the mirror test must now FAIL
    bad_amp = {}
    for pol, sg in ((+1, +1), (-1, -1)):
        m = mk(Q, U, sg)
        m.Ks[0] = m.Ks[0] * 1.3
        m.S = sum(m.Ks)
        _, u, _ = m.drive(pol, Om)
        bad_amp[(pol, sg)] = np.maximum(np.hypot(u[:, 0], u[:, 1]), np.hypot(u[:, 2], u[:, 3]))
    broken = np.abs(bad_amp[(+1, +1)] - bad_amp[(-1, -1)]).max() / sc
    rec("D4", "phonon", "Negative control: broken mirror symmetry", "CONTROL",
        "With one bond stiffened by 30% the D1 test must detect a violation (> 1e-3).", broken, 1e-3, broken > 1e-3,
        note=f"violation detected: {broken:.3g}", mode="min")
    return Om, store, amp


# =============================================================================
# 4. STADIUM CLAIMS
# =============================================================================
def lattice_weyl(E, h, A, P, n=1500):
    k = (np.arange(n) + 0.5) / n * (math.pi / h)
    KX, KY = np.meshgrid(k, k)
    El = (2 - np.cos(KX * h) - np.cos(KY * h)) / h ** 2
    bulk = A / (4 * math.pi ** 2) * 4 * np.mean(El < E) * (math.pi / h) ** 2
    return bulk - P * math.sqrt(2 * E) / (4 * math.pi)


def stadium_claims(Q, U, h, nev, n_shuffle):
    st = Q.Stadium(h)
    H = st.hamiltonian()
    E0 = U.k0 ** 2 / 2
    E, V = eigsh(H, k=nev, sigma=E0, which="LM")
    o = np.argsort(E)
    E, V = E[o], V[:, o]
    N = st.N

    res = np.max(np.linalg.norm(H @ V - V * E, axis=0) / np.abs(E))
    orth = np.abs(V.T @ V - np.eye(nev)).max()
    rec("S1", "stadium", "Eigenpairs solve the Dirichlet problem", "IDENTITY",
        "||H psi_n - E_n psi_n|| / E_n ~ 0 and psi_n orthonormal.", max(res, orth), 1e-9, max(res, orth) <= 1e-9,
        note=f"residual {res:.2e}, orthonormality {orth:.2e}, {N} cells, {nev} states, E in [{E[0]:.1f},{E[-1]:.1f}]")

    A, P = math.pi * st.R ** 2 + 4 * st.R * st.Ls, 2 * math.pi * st.R + 4 * st.Ls
    pred = lattice_weyl(E[-1], h, A, P) - lattice_weyl(E[0], h, A, P)
    rel = abs(pred - nev) / nev
    rec("S2", "stadium", "Weyl law for the level count", "ASYMPTOTIC",
        "N(E) = lattice-corrected A-term minus P sqrt(2E)/(4 pi); count in the computed window must match.",
        rel, 0.05, rel <= 0.05, note=f"counted {nev}, Weyl prediction {pred:.1f} (area {A:.4f}, perimeter {P:.4f})")

    # wavepacket (same definition as qnos_physics.ScarApp.packet)
    x0, y0 = U.launch
    ang = math.radians(U.angle)
    kx, ky = U.k0 * math.cos(ang), U.k0 * math.sin(ang)
    X, Y = st.X[st.mask], st.Y[st.mask]
    psi0 = np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (4 * U.sigma ** 2)) * np.exp(1j * (kx * (X - x0) + ky * (Y - y0)))
    psi0 /= np.linalg.norm(psi0)
    c = V.T @ psi0
    p_raw = np.abs(c) ** 2
    cap = p_raw.sum()
    rec("S3", "stadium", "Bessel inequality / basis adequacy", "BOUND",
        "Captured packet weight sum |c_n|^2 <= 1; values far below 1 mean the window misses part of the packet.",
        cap, 1.0 + 1e-12, cap <= 1.0 + 1e-12, note=f"captured weight {cap:.4f}" + ("" if cap > 0.5 else "  [WARNING: <0.5, results describe the projection only]"))

    # unitarity
    drift = 0.0
    for t in (0.37, 5.1, 123.4, 4000.7):
        drift = max(drift, abs(np.linalg.norm(V @ (c * np.exp(-1j * E * t))) ** 2 - cap) / cap)
    rec("S4", "stadium", "Unitary time evolution", "IDENTITY", "||psi(t)||^2 is constant in t.", drift, 1e-12, drift <= 1e-12)

    # time-average convergence to the diagonal ensemble, with the rigorous bound (D5)
    V2 = V ** 2
    rho_bar = V2 @ p_raw
    dE = E[:, None] - E[None, :]
    iu = ~np.eye(nev, dtype=bool)
    cc = np.abs(c)[:, None] * np.abs(c)[None, :]
    Bsum = np.sum(cc[iu] / np.abs(dE[iu]))

    def avg_density(T):
        with np.errstate(divide="ignore", invalid="ignore"):
            S = (1 - np.exp(-1j * dE * T)) / (1j * dE * T)
        S[~iu] = 1.0
        C = (c[:, None] * np.conj(c)[None, :]) * S
        return np.real(np.sum((V @ C) * V, axis=1))

    # cross-check the closed-form average against Gauss-Legendre quadrature at T=1
    xg, wg = np.polynomial.legendre.leggauss(300)
    tq = 0.5 * (xg + 1.0)
    quad = np.zeros(N)
    for ti, wi in zip(tq, wg):
        quad += 0.5 * wi * np.abs(V @ (c * np.exp(-1j * E * ti))) ** 2
    q_err = np.abs(avg_density(1.0) - quad).sum()
    rec("S5", "stadium", "Closed-form time average = direct quadrature", "IDENTITY",
        "(1/T) int_0^T |psi(t)|^2 dt equals the sinc-weighted double sum (checked at T=1).", q_err, 1e-8, q_err <= 1e-8)

    Ts = np.array([1, 3, 10, 30, 100, 300, 1000.0])
    errs = np.array([np.abs(avg_density(T) - rho_bar).sum() for T in Ts])
    bounds = 2 * Bsum / Ts
    ok = bool(np.all(errs <= bounds * (1 + 1e-9)))
    slope = np.polyfit(np.log(Ts), np.log(errs), 1)[0]
    rec("S6", "stadium", "Convergence to the diagonal ensemble (D5)", "BOUND",
        "|| <rho>_T - rho_bar ||_1 <= (2/T) sum_{n!=m} |c_n c_m|/|E_n-E_m| for every T.",
        float(np.max(errs / bounds)), 1.0, ok,
        note=f"error/bound <= {np.max(errs/bounds):.3f}; measured decay exponent {slope:.2f} (bound has -1); "
             f"min level gap {np.abs(np.diff(E)).min():.3g}")

    # birthmark: is the launch remembered?  STRATIFIED permutation test against an ergodic null.
    # Weights are shuffled only among neighbouring levels (strata of 10), so the packet's energy profile is
    # preserved and only the eigenstate-specific correlation (the "memory") is destroyed.
    p = p_raw / cap
    rho = V2 @ p
    ref = V2.mean(axis=1)                              # equal-weight window ensemble (removes boundary depletion)
    disk = (np.hypot(X - x0, Y - y0) < 0.3)
    stratum = (np.arange(nev) // 10).astype(float)
    refd = ref[disk].mean()

    def stats(pw, rg, n):
        Bs, Ts_ = [], []
        for _ in range(max(1, n // 500)):
            idx = np.argsort(stratum[:, None] + rg.random((nev, 500)) * 0.999, axis=0)
            rn = V2 @ pw[idx]
            Bs.append(rn[disk].mean(axis=0) / refd)
            Ts_.append(0.5 * np.abs(rn - ref[:, None]).sum(axis=0))
        return np.concatenate(Bs), np.concatenate(Ts_)

    B_obs = rho[disk].mean() / refd
    tv_obs = 0.5 * np.abs(rho - ref).sum()
    nB, nT = stats(p, U.rng_perm, n_shuffle)
    pB = (1 + np.sum(nB >= B_obs)) / (1 + nB.size)
    pT = (1 + np.sum(nT >= tv_obs)) / (1 + nT.size)
    zB = (B_obs - nB.mean()) / (nB.std() + 1e-300)
    pmin = min(pB, pT)
    detected = pmin < 0.005                            # Bonferroni: alpha = 0.01 shared by two statistics
    rec("S7", "stadium", "Birthmark: memory of the launch", "STATISTICAL",
        "H0: within energy strata the packet weights are unrelated to the eigenstates' structure. "
        "Two statistics (local enhancement B, global TV distance), Bonferroni alpha=0.01.",
        pmin, 0.005, detected, status="DETECTED" if detected else "NOT DETECTED", mode="max",
        note=f"local enhancement B={B_obs:.2f} (null {nB.mean():.2f}+/-{nB.std():.2f}, z={zB:.1f}, p={pB:.4f}); "
             f"TV={tv_obs:.3f} (null {nT.mean():.3f}+/-{nT.std():.3f}, p={pT:.4f}); N_eff={1/np.sum(p**2):.1f}; "
             f"{nB.size} stratified shuffles")

    # calibration of S7: when H0 is TRUE (pseudo-observed weights = a stratified shuffle) the false-positive
    # rate at alpha=0.05 must be about 5%
    cal_rng = np.random.default_rng(np.random.SeedSequence([U.seed & (2**128 - 1), 3]))
    n_cal, fp = 30, 0
    for _ in range(n_cal):
        pseudo = p[np.argsort(stratum + cal_rng.random(nev) * 0.999)]
        b_ps = (V2 @ pseudo)[disk].mean() / refd
        t_ps = 0.5 * np.abs(V2 @ pseudo - ref).sum()
        cb, ct = stats(p, cal_rng, 500)
        pv = min((1 + np.sum(cb >= b_ps)) / (1 + cb.size), (1 + np.sum(ct >= t_ps)) / (1 + ct.size))
        fp += pv < 0.025                                # two statistics -> Bonferroni at 0.05
    rec("S9", "stadium", "Calibration of the birthmark test (type-I error)", "CONTROL",
        "Under a true null the test must not cry wolf: false-positive rate at alpha=0.05 stays <= 20% "
        "(binomial 99% limit for 30 trials).", fp / n_cal, 0.20, fp / n_cal <= 0.20,
        note=f"{fp}/{n_cal} false positives")

    # discretisation convergence (lowest eigenvalues, Richardson-style)
    hs = [h, h / 2, h / 4]
    lows = []
    for hh in hs:
        s2 = Q.Stadium(hh)
        lows.append(np.sort(eigsh(s2.hamiltonian(), k=6, sigma=0.0, which="LM")[0]))
    d12, d23 = np.abs(lows[0] - lows[1]), np.abs(lows[1] - lows[2])
    order = np.median(np.log2(d12 / np.maximum(d23, 1e-300)))
    good = int(np.sum(d23 < d12))
    rec("S8", "stadium", "Grid convergence (6 lowest levels)", "ASYMPTOTIC",
        "Successive refinements h, h/2, h/4 change the eigenvalues by less and less.", float(good), 5.0, good >= 5,
        note=f"{good}/6 levels converge; median observed order {order:.2f}; E1 = {lows[2][0]:.5f} (h/4)", mode="min")

    return dict(st=st, E=E, V=V, rho=rho, ref=ref, Ts=Ts, errs=errs, bounds=bounds, nB=nB, B_obs=B_obs, pB=pB,
                disk=disk, cap=cap)


# =============================================================================
# 5. REPORT AND FIGURE
# =============================================================================
def certificate(U, params_hash, phys_path, phys_hash, runtime, args):
    counted = [r for r in RESULTS if r["status"] in ("PASS", "FAIL")]
    fails = [r["id"] for r in counted if r["status"] == "FAIL"]
    core = {"text_sha256": U.digest, "params": U.params(), "verdicts": {r["id"]: r["status"] for r in RESULTS}}
    digest = hashlib.sha256(json.dumps(core, sort_keys=True, default=float).encode()).hexdigest()
    return dict(program="qnos_proofs.py", version=VERSION, text=U.text, text_sha256=U.digest, params=U.params(),
                physics_module=os.path.basename(phys_path), physics_sha256=phys_hash,
                environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                                 matplotlib=matplotlib.__version__, platform=platform.platform()),
                settings=dict(h=args.h, nev=args.nev, shuffles=args.shuffles), runtime_s=round(runtime, 1),
                results=RESULTS, passed=sum(r["status"] == "PASS" for r in RESULTS), failed=fails,
                not_applicable=[r["id"] for r in RESULTS if r["status"] == "N/A"],
                statistical={r["id"]: r["status"] for r in RESULTS if r["kind"] == "STATISTICAL"},
                certificate_digest=digest)


def write_text_report(cert, path):
    L = []
    L.append("QNOS VERIFICATION CERTIFICATE")
    L.append("=" * 78)
    L.append(f"text            : {cert['text']!r}")
    L.append(f"sha256(text)    : {cert['text_sha256']}")
    L.append(f"physics module  : {cert['physics_module']}  sha256 {cert['physics_sha256'][:16]}...")
    L.append(f"environment     : python {cert['environment']['python']}, numpy {cert['environment']['numpy']}, "
             f"scipy {cert['environment']['scipy']}")
    L.append(f"settings        : h={cert['settings']['h']}, states={cert['settings']['nev']}, shuffles={cert['settings']['shuffles']}")
    L.append("")
    L.append("Configuration selected by the text")
    for k, v in cert["params"].items():
        L.append(f"  {k:8s} = {v if not isinstance(v, float) else round(v, 6)}")
    L.append("")
    for grp in ("phonon", "stadium"):
        L.append(f"[{grp.upper()}]")
        for r in (x for x in cert["results"] if x["group"] == grp):
            rel = "<=" if r["mode"] == "max" else ">="
            L.append(f"  {r['id']:3s} {r['status']:12s} {r['kind']:11s} {r['title']}")
            L.append(f"        claim   : {r['claim']}")
            L.append(f"        measured: {r['measured']:.4g}  (criterion {rel} {r['tol']:.4g})")
            if r["note"]:
                L.append(f"        detail  : {r['note']}")
        L.append("")
    L.append("SUMMARY")
    L.append(f"  passed {cert['passed']}, failed {cert['failed'] or 'none'}, not applicable {cert['not_applicable'] or 'none'}")
    L.append(f"  statistical verdicts: {cert['statistical']}")
    L.append(f"  certificate digest: {cert['certificate_digest']}")
    L.append("")
    L.append("INTERPRETATION LIMITS")
    L.append("  * These are numerical verifications at one text-selected parameter point, not formal proofs.")
    L.append("  * The phonon model is a minimal lattice-dynamics model, not an ab-initio material calculation.")
    L.append("  * 'DETECTED' / 'NOT DETECTED' is a hypothesis-test outcome; both are valid results.")
    L.append("  * Run several different texts: IDENTITY / CLOSED-FORM / BOUND claims must pass for every one.")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return "\n".join(L)


def make_figure(Q, U, cert, Om, store, amp, sd, path):
    s = U.neel
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
    # (a) dispersion cut through Gamma showing non-reciprocity
    m = mk(Q, U, s)
    Kmax = 4 * math.pi / 3
    kx = np.linspace(-Kmax, Kmax, 241)
    sols = [m.solve(np.array([k, 0.0])) for k in kx]
    w = np.array([x[0] for x in sols]); l = np.array([x[2] for x in sols])
    for j in range(4):
        sc = ax[0, 0].scatter(kx, w[:, j], c=l[:, j], cmap="coolwarm", vmin=-1, vmax=1, s=5)
    cf = closed_form_gamma(U)
    ax[0, 0].plot([0, 0], [cf["ccw"], cf["cw"]], "k*", ms=9, label="closed form (D1)")
    ax[0, 0].set_xlabel("k_x  (cut through Gamma)"); ax[0, 0].set_ylabel("frequency")
    ax[0, 0].set_title(f"(a) Bands, colour = phonon angular momentum\nNeel {s:+d}; asymmetry under k->-k = T-breaking")
    ax[0, 0].legend(fontsize=8); plt.colorbar(sc, ax=ax[0, 0], label="ell")
    # (b) driven response
    t = store[(+1, s)][0]
    ax[0, 1].plot(t, amp[(+1, s)], label="sigma+ pump", lw=1.6)
    ax[0, 1].plot(t, amp[(-1, s)], label="sigma- pump", lw=1.6)
    ax[0, 1].set_xlabel("t"); ax[0, 1].set_ylabel("max atomic amplitude")
    d3 = next(r for r in cert["results"] if r["id"] == "D3")
    ax[0, 1].set_title(f"(b) Circular dichroism at w={Om:.4f}\nhidden order {s:+d}; D3: {d3['status']}")
    ax[0, 1].legend()
    # (c) residual vs tolerance
    rows = [r for r in cert["results"] if r["status"] in ("PASS", "FAIL") and r["kind"] not in ("STATISTICAL", "CONTROL")
            and r["mode"] == "max"]
    ids = [r["id"] for r in rows]
    meas = [max(r["measured"], 1e-16) for r in rows]   # exact zeros are drawn at the 1e-16 floor
    tol = [r["tol"] for r in rows]
    y = np.arange(len(rows))
    ax[0, 2].barh(y, meas, left=1e-17, color=["tab:green" if r["status"] == "PASS" else "tab:red" for r in rows], alpha=0.8)
    ax[0, 2].scatter(tol, y, marker="|", color="k", s=300)
    ax[0, 2].set_xscale("log"); ax[0, 2].set_yticks(y); ax[0, 2].set_yticklabels(ids)
    ax[0, 2].set_xlim(1e-17, 10); ax[0, 2].invert_yaxis()
    ax[0, 2].set_title("(c) Residual of each exact claim\n(black tick = tolerance; zeros drawn at 1e-16)", fontsize=10)
    ax[0, 2].set_xlabel("measured residual (log)")
    # (d) bound
    ax[1, 0].loglog(sd["Ts"], sd["errs"], "o-", label="measured L1 error")
    ax[1, 0].loglog(sd["Ts"], sd["bounds"], "s--", label="rigorous bound (D5)")
    ax[1, 0].set_xlabel("averaging time T"); ax[1, 0].set_ylabel("|| <rho>_T - rho_bar ||_1")
    ax[1, 0].set_title("(d) Time average converges to diagonal ensemble"); ax[1, 0].legend()
    # (e) permutation null
    ax[1, 1].hist(sd["nB"], bins=40, color="0.7", label="shuffled-weight null")
    ax[1, 1].axvline(sd["B_obs"], color="crimson", lw=2, label=f"observed B={sd['B_obs']:.2f}")
    ax[1, 1].set_xlabel("enhancement near launch point"); ax[1, 1].set_ylabel("count")
    ax[1, 1].set_title(f"(e) Birthmark test: p = {sd['pB']:.4f}"); ax[1, 1].legend()
    # (f) excess map
    st = sd["st"]
    exc = st.to_grid((sd["rho"] - sd["ref"]) / sd["ref"].mean())
    lim = np.nanpercentile(np.abs(exc), 99)
    im = ax[1, 2].imshow(exc, extent=[st.xs[0], st.xs[-1], st.ys[0], st.ys[-1]], origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax[1, 2].plot(*U.launch, "k+", ms=14, mew=2)
    ax[1, 2].set_title("(f) Excess time-averaged density\nvs ergodic reference (+ = launch)", fontsize=10)
    plt.colorbar(im, ax=ax[1, 2], label="(rho_bar - rho_ref)/<rho_ref>", fraction=0.04, pad=0.03)
    fig.suptitle(f"QNOS verification for text {U.text[:50]!r}   |   {cert['passed']} claims passed, "
                 f"failed: {cert['failed'] or 'none'}   |   digest {cert['certificate_digest'][:16]}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=140)
    plt.close(fig)


# =============================================================================
# 6. DRIVER
# =============================================================================
def run(text, args):
    RESULTS.clear()
    t0 = time.time()
    Q, ppath, phash = load_physics(args.physics)
    U = SeededUniverse(text)
    print(f"text sha256 {U.digest[:16]}...  ->  mB/mA={U.mB:.3f} g={U.g:.3f} Neel {U.neel:+d} "
          f"k0={U.k0:.2f} launch=({U.launch[0]:.2f},{U.launch[1]:.2f}) angle={U.angle:.0f}")
    print("phonon claims ..."); phonon_static_claims(Q, U)
    Om, store, amp = phonon_drive_claims(Q, U)
    print("stadium claims ..."); sd = stadium_claims(Q, U, args.h, args.nev, args.shuffles)
    cert = certificate(U, None, ppath, phash, time.time() - t0, args)
    os.makedirs(args.out, exist_ok=True)
    tag = U.digest[:10]
    with open(os.path.join(args.out, f"certificate_{tag}.json"), "w") as f:
        json.dump(cert, f, indent=2, default=float)
    print(write_text_report(cert, os.path.join(args.out, f"certificate_{tag}.txt")))
    make_figure(Q, U, cert, Om, store, amp, sd, os.path.join(args.out, f"certificate_{tag}.png"))
    print(f"\nwrote certificate_{tag}.json / .txt / .png in {args.out}/")
    return cert


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="?", help="text that seeds the physical configuration")
    ap.add_argument("--out", default="qnos_proofs_out")
    ap.add_argument("--h", type=float, default=0.03, help="billiard grid spacing")
    ap.add_argument("--nev", type=int, default=220, help="number of billiard eigenstates")
    ap.add_argument("--shuffles", type=int, default=2000, help="permutation-test shuffles (multiple of 500)")
    ap.add_argument("--quick", action="store_true", help="coarser grid, fewer states and shuffles")
    ap.add_argument("--physics", help="path to qnos_physics20262.py if it is not next to this script")
    args = ap.parse_args()
    if args.quick:
        args.h, args.nev, args.shuffles = 0.04, 120, 500
    text = args.text
    if not text:
        text = input("> Enter text to seed the physics: ").strip()
    if not text:
        sys.exit("Text required.")
    cert = run(text, args)
    sys.exit(1 if cert["failed"] else 0)


if __name__ == "__main__":
    main()
