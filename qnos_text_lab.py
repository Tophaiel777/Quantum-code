#!/usr/bin/env python3
"""
QNOS TEXT LAB -- text in, physics out, data certificate written
================================================================
Requires: numpy, scipy, matplotlib  and  qnos_physics20262.py (or qnos_physics.py) in the same folder.

    python qnos_text_lab.py "any text you like"
    python qnos_text_lab.py "any text" --out results --quick --show
    python qnos_text_lab.py                       # interactive prompt, like ImplicitGenerator_Lab

WHAT THE TEXT DOES
------------------
Same idea as ImplicitGenerator_Lab: the text is the seed and nothing else is typed in by hand.
The text is hashed (SHA-256) and the hash seeds a NumPy generator, which fixes ONE physical configuration
of the QNOS models:
    * honeycomb phonon lattice : sublattice masses, spring constants, spin-phonon coupling g, Neel sign
    * stadium billiard         : wavepacket launch point, direction, wavenumber, width
The program then runs the physics for that configuration and MEASURES things. Nothing is pass/fail here.
Every number in the certificate is a measurement produced by the text-selected system. Same text -> same
configuration -> same numbers, on every machine (up to floating-point noise in the last digits).

OUTPUTS (in --out, tagged with the first 10 hex digits of the text hash)
-------
    qnos_<tag>.txt   the scientific data certificate (human readable)
    qnos_<tag>.json  the same data, machine readable
    qnos_<tag>.png   a 6-panel figure of the same data (use --show to also open a window)

WHAT IS MEASURED
----------------
PHONON (honeycomb, magnetic order breaks time reversal)
    Gamma-point optical doublet (numeric and from the analytic formula), CCW/CW splitting, angular momenta,
    band extrema, non-reciprocity max|w(k)-w(-k)|, trace sum, circularly polarised drive: peak amplitudes for
    sigma+ / sigma-, dichroic ratio and the Neel sign that the dichroism points to.
STADIUM (Dirichlet billiard, eigenstates near the packet energy)
    level count vs Weyl law, mean level spacing, spacing-ratio statistic <r> (Poisson 0.386 .. GOE 0.536),
    eigenstate participation, wavepacket energy / spectral width / captured weight / effective number of
    states, long-time density, and the "birthmark" (does the long-time density remember the launch point?)
    with a stratified permutation test.

DERIVATIONS THE ANALYTIC COLUMNS USE
------------------------------------
(D1) Gamma point: Phi0 = [[S,-S],[-S,S]], S = kappa*I, kappa = 3/2 (phi_l+phi_t).
         w_CCW,CW = ( +/- g s (mA-mB) + D ) / (2 mA mB),  D = sqrt( g^2 (mA-mB)^2 + 4 mA mB (kappa (mA+mB) + g^2) )
         w_CCW - w_CW = g s (1/mB - 1/mA)
(D3) Trace sum: sum_j [ w_j^2(k) + w_j^2(-k) ] = 2 tr(S)(1/mA+1/mB) + 2 g^2 (1/mA^2 + 1/mB^2)
"""
import argparse, hashlib, importlib.util, json, math, os, platform, sys, time

import numpy as np
import scipy
from scipy.sparse.linalg import eigsh

VERSION = "1.0"


# =============================================================================
# 0. LOAD THE QNOS PHYSICS MODULE
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
    """Deterministic map  text -> physical configuration."""

    def __init__(self, text, R=1.0, Ls=1.0):
        self.text = text
        self.digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        self.seed = int(self.digest, 16)
        r = np.random.default_rng(np.random.SeedSequence(self.seed))
        U = lambda a, b: float(r.uniform(a, b))
        # phonon lattice
        self.mA = 1.0
        self.mB = U(1.6, 3.0)
        self.phi_l = U(0.8, 1.4)
        self.phi_t = self.phi_l * U(0.25, 0.5)
        self.g = U(0.25, 0.45)
        self.neel = int(r.choice([-1, 1]))
        self.kappa = 1.5 * (self.phi_l + self.phi_t)
        # stadium wavepacket
        self.k0, self.sigma, self.angle = U(15.0, 20.0), U(0.14, 0.22), U(0.0, 360.0)
        for _ in range(100000):
            x, y = U(-1.7, 1.7), U(-0.7, 0.7)
            if R - math.hypot(x - min(max(x, -Ls), Ls), y) > 3 * self.sigma:
                break
        self.launch = (x, y)
        # independent streams
        self.rng_k = np.random.default_rng(np.random.SeedSequence([self.seed & (2**128 - 1), 1]))
        self.rng_perm = np.random.default_rng(np.random.SeedSequence([self.seed & (2**128 - 1), 2]))

    def params(self):
        keys = ("mA", "mB", "phi_l", "phi_t", "g", "neel", "k0", "sigma", "angle", "launch")
        return {k: getattr(self, k) for k in keys}

    def text_signature(self):
        b = self.text.encode("utf-8")
        bits = "".join(format(c, "08b") for c in b)
        return {"characters": len(self.text), "bytes": len(b), "bits": len(bits),
                "fraction_of_one_bits": bits.count("1") / len(bits), "sha256": self.digest}


# =============================================================================
# 2. DATA RECORDS
# =============================================================================
DATA = []   # list of dicts: section, key, value, unit, note


def put(section, key, value, unit="", note=""):
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    DATA.append(dict(section=section, key=key, value=value, unit=unit, note=note))
    return value


def closed_form_gamma(U, s=None):
    s = U.neel if s is None else s
    mA, mB = U.mA, U.mB
    D = math.sqrt(U.g ** 2 * (mA - mB) ** 2 + 4 * mA * mB * (U.kappa * (mA + mB) + U.g ** 2))
    return dict(ccw=(U.g * s * (mA - mB) + D) / (2 * mA * mB), cw=(-U.g * s * (mA - mB) + D) / (2 * mA * mB))


def mk(Q, U, s, g=None, mB=None):
    return Q.PhononModel(mA=U.mA, mB=U.mB if mB is None else mB, phi_l=U.phi_l, phi_t=U.phi_t,
                         g=U.g if g is None else g, neel=s)


# =============================================================================
# 3. PHONON MEASUREMENTS
# =============================================================================
def measure_phonon(Q, U, nk=24):
    sec = "PHONON"
    s = U.neel
    m = mk(Q, U, s)

    # --- Gamma point
    w, _, l = m.solve(np.zeros(2))
    put(sec, "gamma_frequencies", [float(x) for x in w], "", "all four positive branches at k=0, ascending")
    put(sec, "gamma_angular_momenta", [float(x) for x in l], "hbar", "phonon angular momentum of each branch")
    up = [j for j in (2, 3) if l[j] > 0]
    dn = [j for j in (2, 3) if l[j] < 0]
    cf = closed_form_gamma(U)
    if len(up) == 1 and len(dn) == 1:
        wc, ww = float(w[up[0]]), float(w[dn[0]])
        put(sec, "optical_CCW_frequency", wc, "", f"analytic (D1): {cf['ccw']:.8f}")
        put(sec, "optical_CW_frequency", ww, "", f"analytic (D1): {cf['cw']:.8f}")
        put(sec, "optical_splitting_CCW_minus_CW", wc - ww, "", f"analytic g s (1/mB-1/mA) = {U.g * s * (1 / U.mB - 1 / U.mA):+.8f}")
        put(sec, "numeric_vs_analytic_error", max(abs(wc - cf['ccw']), abs(ww - cf['cw'])), "", "max abs difference, Gamma doublet")
    else:
        put(sec, "optical_doublet", "not cleanly circular", "", "could not assign one CCW and one CW optical mode")
    put(sec, "splitting_relative_to_mean_optical", abs(U.g * (1 / U.mB - 1 / U.mA)) / max(cf['ccw'], 1e-12), "",
        "|splitting| / CCW frequency")

    # --- bands along a cut through Gamma (kx axis)
    Kmax = 4 * math.pi / 3
    kx = np.linspace(-Kmax, Kmax, 241)
    sols = [m.solve(np.array([k, 0.0])) for k in kx]
    W = np.array([x[0] for x in sols])
    L = np.array([x[2] for x in sols])
    put(sec, "band_min_frequency", float(W.min()), "", "over the kx cut")
    put(sec, "band_max_frequency", float(W.max()), "", "over the kx cut")
    # w(k) - w(-k): the cut is symmetric, so reverse the k axis
    nr_cut = float(np.abs(W - W[::-1]).max())
    put(sec, "nonreciprocity_max_on_cut", nr_cut, "", "max |w(k) - w(-k)| on the kx cut (0 if time reversal intact)")

    # --- random k points in the Brillouin cell
    u, v = U.rng_k.random((2, nk))
    ks = u[:, None] * Q.PhononModel.b1 + v[:, None] * Q.PhononModel.b2
    nr = max(float(np.abs(m.solve(k)[0] - m.solve(-k)[0]).max()) for k in ks)
    put(sec, "nonreciprocity_max_random_k", nr, "", f"{nk} text-seeded random k-points")
    rhs = 2 * np.trace(m.S) * (1 / U.mA + 1 / U.mB) + 2 * U.g ** 2 * (1 / U.mA ** 2 + 1 / U.mB ** 2)
    tsum = [float((m.solve(k)[0] ** 2).sum() + (m.solve(-k)[0] ** 2).sum()) for k in ks]
    put(sec, "trace_sum_measured_mean", float(np.mean(tsum)), "", f"analytic (D3): {rhs:.8f}")
    put(sec, "trace_sum_spread", float(np.ptp(tsum)), "", "max-min over k (0 if k-independent)")
    ang = [float(np.abs(m.solve(k)[2]).max()) for k in ks]
    put(sec, "max_abs_angular_momentum", max(ang), "hbar", "over the random k-points")

    # --- circularly polarised drive
    Om = closed_form_gamma(U, s=+1)["ccw"]
    amp, store = {}, {}
    for pol in (+1, -1):
        t, uu, _ = mk(Q, U, s).drive(pol, Om)
        amp[pol] = np.maximum(np.hypot(uu[:, 0], uu[:, 1]), np.hypot(uu[:, 2], uu[:, 3]))
        store[pol] = (t, uu)
    pkp, pkm = float(amp[+1].max()), float(amp[-1].max())
    ratio = max(pkp, pkm) / max(min(pkp, pkm), 1e-300)
    decoded = +1 if pkp > pkm else -1
    put(sec, "pump_frequency", Om, "", "analytic CCW Gamma branch of the s=+1 reference")
    put(sec, "peak_amplitude_sigma_plus", pkp, "arb.", "")
    put(sec, "peak_amplitude_sigma_minus", pkm, "arb.", "")
    put(sec, "dichroic_ratio", ratio, "", "larger / smaller peak amplitude")
    put(sec, "dichroism_points_to_neel_sign", decoded, "", "sign of the order favoured by matched helicity")
    put(sec, "seeded_neel_sign", s, "", "the hidden order chosen by the text")
    delta = abs(U.g * (1 / U.mB - 1 / U.mA))
    put(sec, "readout_agrees_with_seed", bool(decoded == s), "",
        "reliable" if delta >= 0.08 else f"splitting {delta:.3f} is small against the pulse bandwidth; treat as indicative")
    return dict(Om=Om, store=store, amp=amp, kx=kx, W=W, L=L, cf=cf)


# =============================================================================
# 4. STADIUM MEASUREMENTS
# =============================================================================
def lattice_weyl(E, h, A, P, n=1500):
    k = (np.arange(n) + 0.5) / n * (math.pi / h)
    KX, KY = np.meshgrid(k, k)
    El = (2 - np.cos(KX * h) - np.cos(KY * h)) / h ** 2
    bulk = A / (4 * math.pi ** 2) * 4 * np.mean(El < E) * (math.pi / h) ** 2
    return bulk - P * math.sqrt(2 * E) / (4 * math.pi)


def measure_stadium(Q, U, h, nev, n_shuffle):
    sec = "STADIUM"
    st = Q.Stadium(h)
    H = st.hamiltonian()
    E0 = U.k0 ** 2 / 2
    E, V = eigsh(H, k=nev, sigma=E0, which="LM")
    o = np.argsort(E)
    E, V = E[o], V[:, o]
    N = st.N

    put(sec, "grid_spacing_h", h, "", f"{N} interior cells")
    put(sec, "states_computed", nev, "", "eigenstates nearest the packet energy")
    put(sec, "target_energy_k0^2/2", E0, "", f"k0 = {U.k0:.4f}")
    put(sec, "energy_window", [float(E[0]), float(E[-1])], "", "lowest and highest computed eigenvalue")
    put(sec, "eigen_residual_max", float(np.max(np.linalg.norm(H @ V - V * E, axis=0) / np.abs(E))), "", "||H psi - E psi|| / E")

    A, P = math.pi * st.R ** 2 + 4 * st.R * st.Ls, 2 * math.pi * st.R + 4 * st.Ls
    pred = lattice_weyl(E[-1], h, A, P) - lattice_weyl(E[0], h, A, P)
    put(sec, "weyl_predicted_count", float(pred), "", f"counted {nev}; area {A:.4f}, perimeter {P:.4f}")
    put(sec, "weyl_relative_difference", abs(pred - nev) / nev, "", "|Weyl - counted| / counted")

    gaps = np.diff(E)
    s_norm = gaps / gaps.mean()
    r = np.minimum(gaps[:-1], gaps[1:]) / np.maximum(np.maximum(gaps[:-1], gaps[1:]), 1e-300)
    put(sec, "mean_level_spacing", float(gaps.mean()), "", "")
    put(sec, "min_level_spacing", float(gaps.min()), "", "")
    put(sec, "spacing_ratio_mean_r", float(r.mean()), "", "Poisson 0.386, GOE 0.536 (stadium has unresolved symmetry sectors, so expect between)")
    put(sec, "spacing_variance_normalised", float(s_norm.var()), "", "Poisson 1.0, GOE ~0.286")

    V2 = V ** 2
    ipr = N * np.sum(V2 ** 2, axis=0)
    put(sec, "eigenstate_IPR_mean", float(ipr.mean()), "", "N*sum(psi^4); real random wave (GOE) ~ 3, localised states larger")
    put(sec, "eigenstate_IPR_max", float(ipr.max()), "", "most localised state in the window (scar candidate)")
    put(sec, "most_localised_state_energy", float(E[int(np.argmax(ipr))]), "", "")

    # --- wavepacket
    x0, y0 = U.launch
    ang = math.radians(U.angle)
    kx, ky = U.k0 * math.cos(ang), U.k0 * math.sin(ang)
    X, Y = st.X[st.mask], st.Y[st.mask]
    psi0 = np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (4 * U.sigma ** 2)) * np.exp(1j * (kx * (X - x0) + ky * (Y - y0)))
    psi0 /= np.linalg.norm(psi0)
    c = V.T @ psi0
    p_raw = np.abs(c) ** 2
    cap = float(p_raw.sum())
    p = p_raw / cap
    Em = float(np.sum(p * E))
    Ew = float(math.sqrt(max(np.sum(p * E ** 2) - Em ** 2, 0.0)))
    put(sec, "packet_launch_point", [x0, y0], "", f"angle {U.angle:.2f} deg, sigma {U.sigma:.4f}")
    put(sec, "packet_captured_weight", cap, "", "fraction of the packet inside the computed window" + ("" if cap > 0.5 else "  [low: results describe the projection only]"))
    put(sec, "packet_mean_energy", Em, "", "")
    put(sec, "packet_energy_width", Ew, "", "")
    put(sec, "packet_effective_states", float(1 / np.sum(p ** 2)), "", "1/sum p^2")
    put(sec, "packet_spectral_entropy", float(-np.sum(p[p > 0] * np.log(p[p > 0]))), "nats", "")

    # --- long-time density (diagonal ensemble) and birthmark
    rho = V2 @ p
    ref = V2.mean(axis=1)
    disk = np.hypot(X - x0, Y - y0) < 0.3
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

    B_obs = float(rho[disk].mean() / refd)
    tv_obs = float(0.5 * np.abs(rho - ref).sum())
    nB, nT = stats(p, U.rng_perm, n_shuffle)
    pB = float((1 + np.sum(nB >= B_obs)) / (1 + nB.size))
    pT = float((1 + np.sum(nT >= tv_obs)) / (1 + nT.size))
    zB = float((B_obs - nB.mean()) / (nB.std() + 1e-300))
    pmin = min(pB, pT)
    put(sec, "birthmark_local_enhancement_B", B_obs, "", f"long-time density within 0.3 of launch / ergodic reference; null {nB.mean():.3f} +/- {nB.std():.3f}")
    put(sec, "birthmark_z_score", zB, "sigma", "against the stratified shuffle null")
    put(sec, "birthmark_p_local", pB, "", f"{nB.size} stratified shuffles")
    put(sec, "birthmark_TV_distance", tv_obs, "", f"total-variation distance from ergodic reference; null {nT.mean():.3f} +/- {nT.std():.3f}")
    put(sec, "birthmark_p_global", pT, "", "")
    put(sec, "birthmark_verdict", "DETECTED" if pmin < 0.005 else "NOT DETECTED", "",
        "min p < 0.005 (Bonferroni alpha 0.01 over two statistics); both outcomes are valid results")
    return dict(st=st, E=E, V=V, rho=rho, ref=ref, nB=nB, B_obs=B_obs, pB=pB, p=p, s_norm=s_norm, disk=disk)


# =============================================================================
# 5. CERTIFICATE
# =============================================================================
def _fmt(v):
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:.8g}"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    return str(v)


def _round_sig(v, n=6):
    if isinstance(v, float):
        return float(f"{v:.{n}g}")
    if isinstance(v, (list, tuple)):
        return [_round_sig(x, n) for x in v]
    return v


def build_certificate(U, phys_path, phys_hash, runtime, args):
    config_digest = hashlib.sha256(json.dumps({"text_sha256": U.digest, "params": U.params()},
                                              sort_keys=True, default=float).encode()).hexdigest()
    rounded = [(d["section"], d["key"], _round_sig(d["value"])) for d in DATA]
    data_digest = hashlib.sha256(json.dumps(rounded, default=str).encode()).hexdigest()
    return dict(program="qnos_text_lab.py", version=VERSION, text=U.text, text_signature=U.text_signature(),
                params=U.params(), physics_module=os.path.basename(phys_path), physics_sha256=phys_hash,
                environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                                 platform=platform.platform()),
                settings=dict(h=args.h, nev=args.nev, shuffles=args.shuffles), runtime_s=round(runtime, 1),
                data=DATA, config_digest=config_digest, data_digest=data_digest)


def write_text_report(cert, path):
    L = []
    L.append("QNOS TEXT LAB - DATA CERTIFICATE")
    L.append("=" * 78)
    L.append(f"text            : {cert['text']!r}")
    sig = cert["text_signature"]
    L.append(f"text signature  : {sig['characters']} chars, {sig['bytes']} bytes, {sig['bits']} bits, "
             f"{sig['fraction_of_one_bits']:.4f} of bits are 1")
    L.append(f"sha256(text)    : {sig['sha256']}")
    L.append(f"physics module  : {cert['physics_module']}  sha256 {cert['physics_sha256'][:16]}...")
    L.append(f"environment     : python {cert['environment']['python']}, numpy {cert['environment']['numpy']}, "
             f"scipy {cert['environment']['scipy']}")
    L.append(f"settings        : h={cert['settings']['h']}, states={cert['settings']['nev']}, "
             f"shuffles={cert['settings']['shuffles']}   (runtime {cert['runtime_s']} s)")
    L.append("")
    L.append("CONFIGURATION SELECTED BY THE TEXT")
    for k, v in cert["params"].items():
        L.append(f"  {k:8s} = {_fmt(v)}")
    for sec in ("PHONON", "STADIUM"):
        L.append("")
        L.append(f"[{sec}]")
        for d in (x for x in cert["data"] if x["section"] == sec):
            unit = f" {d['unit']}" if d["unit"] else ""
            L.append(f"  {d['key']:34s} {_fmt(d['value'])}{unit}")
            if d["note"]:
                L.append(f"  {'':34s}   ({d['note']})")
    L.append("")
    L.append("REPRODUCIBILITY")
    L.append(f"  configuration digest : {cert['config_digest']}")
    L.append(f"  data digest (6 sig.) : {cert['data_digest']}")
    L.append("  The configuration digest depends only on the text. The data digest also depends on the")
    L.append("  physics module and numerical libraries; rerun the same text with the same settings to compare.")
    L.append("")
    L.append("INTERPRETATION LIMITS")
    L.append("  * Every value is a measurement at the single parameter point this text selected.")
    L.append("  * The phonon model is a minimal lattice-dynamics model, not an ab-initio material calculation.")
    L.append("  * The birthmark verdict is a hypothesis-test outcome; DETECTED and NOT DETECTED are both valid.")
    L.append("  * Analytic columns (D1, D3) are shown next to numeric values for comparison, not as pass/fail tests.")
    txt = "\n".join(L)
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)
    return txt


# =============================================================================
# 6. FIGURE
# =============================================================================
def make_figure(Q, U, cert, ph, sd, path, show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = U.neel
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))

    # (a) bands
    for j in range(4):
        sc = ax[0, 0].scatter(ph["kx"], ph["W"][:, j], c=ph["L"][:, j], cmap="coolwarm", vmin=-1, vmax=1, s=5)
    ax[0, 0].plot([0, 0], [ph["cf"]["ccw"], ph["cf"]["cw"]], "k*", ms=9, label="analytic Gamma doublet")
    ax[0, 0].set_xlabel("k_x (cut through Gamma)"); ax[0, 0].set_ylabel("frequency")
    ax[0, 0].set_title(f"(a) Phonon bands, colour = angular momentum\nNeel {s:+d}; asymmetry under k->-k = T-breaking")
    ax[0, 0].legend(fontsize=8); plt.colorbar(sc, ax=ax[0, 0], label="ell")

    # (b) driven response
    t = ph["store"][+1][0]
    ax[0, 1].plot(t, ph["amp"][+1], label="sigma+ pump", lw=1.6)
    ax[0, 1].plot(t, ph["amp"][-1], label="sigma- pump", lw=1.6)
    ax[0, 1].set_xlabel("t"); ax[0, 1].set_ylabel("max atomic amplitude")
    ax[0, 1].set_title(f"(b) Circular dichroism at w={ph['Om']:.4f}\nseeded order {s:+d}")
    ax[0, 1].legend()

    # (c) level-spacing distribution
    sn = sd["s_norm"]
    ax[0, 2].hist(sn, bins=25, range=(0, 4), density=True, color="0.75", label="computed spacings")
    xs = np.linspace(0, 4, 200)
    ax[0, 2].plot(xs, np.exp(-xs), "b--", label="Poisson")
    ax[0, 2].plot(xs, math.pi / 2 * xs * np.exp(-math.pi * xs ** 2 / 4), "r-", label="GOE Wigner")
    ax[0, 2].set_xlabel("spacing / mean spacing"); ax[0, 2].set_ylabel("density")
    ax[0, 2].set_title("(c) Stadium level-spacing distribution"); ax[0, 2].legend(fontsize=8)

    # (d) packet spectrum
    ax[1, 0].vlines(sd["E"], 0, sd["p"], color="tab:purple", lw=1)
    ax[1, 0].set_xlabel("eigenenergy"); ax[1, 0].set_ylabel("packet weight  |c_n|^2 / captured")
    ax[1, 0].set_title("(d) Wavepacket spectrum in the stadium eigenbasis")

    # (e) permutation null
    ax[1, 1].hist(sd["nB"], bins=40, color="0.7", label="shuffled-weight null")
    ax[1, 1].axvline(sd["B_obs"], color="crimson", lw=2, label=f"observed B={sd['B_obs']:.2f}")
    ax[1, 1].set_xlabel("enhancement near launch point"); ax[1, 1].set_ylabel("count")
    ax[1, 1].set_title(f"(e) Birthmark test: p = {sd['pB']:.4f}"); ax[1, 1].legend()

    # (f) excess density map
    st = sd["st"]
    exc = st.to_grid((sd["rho"] - sd["ref"]) / sd["ref"].mean())
    lim = np.nanpercentile(np.abs(exc), 99)
    im = ax[1, 2].imshow(exc, extent=[st.xs[0], st.xs[-1], st.ys[0], st.ys[-1]], origin="lower",
                         cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax[1, 2].plot(*U.launch, "k+", ms=14, mew=2)
    ax[1, 2].set_title("(f) Excess long-time density vs ergodic reference\n(+ marks the launch point)", fontsize=10)
    plt.colorbar(im, ax=ax[1, 2], fraction=0.04, pad=0.03)

    shown = U.text if len(U.text) < 50 else U.text[:50] + "..."
    fig.suptitle(f"QNOS text lab  |  input: {shown!r}  |  config {cert['config_digest'][:16]}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=140)
    if show:
        plt.show()
    plt.close(fig)


# =============================================================================
# 7. DRIVER
# =============================================================================
def run(text, args):
    DATA.clear()
    t0 = time.time()
    Q, ppath, phash = load_physics(args.physics)
    U = SeededUniverse(text)
    print(f"text sha256 {U.digest[:16]}...  ->  mB/mA={U.mB:.3f} g={U.g:.3f} Neel {U.neel:+d} "
          f"k0={U.k0:.2f} launch=({U.launch[0]:.2f},{U.launch[1]:.2f}) angle={U.angle:.0f}")
    print("phonon measurements ...")
    ph = measure_phonon(Q, U)
    print("stadium measurements ...")
    sd = measure_stadium(Q, U, args.h, args.nev, args.shuffles)
    cert = build_certificate(U, ppath, phash, time.time() - t0, args)
    os.makedirs(args.out, exist_ok=True)
    tag = U.digest[:10]
    with open(os.path.join(args.out, f"qnos_{tag}.json"), "w") as f:
        json.dump(cert, f, indent=2, default=str)
    print(write_text_report(cert, os.path.join(args.out, f"qnos_{tag}.txt")))
    make_figure(Q, U, cert, ph, sd, os.path.join(args.out, f"qnos_{tag}.png"), show=args.show)
    print(f"\nwrote qnos_{tag}.txt / .json / .png in {args.out}/")
    return cert


def interactive(args):
    print("\n" + "=" * 50)
    print(" QNOS TEXT LAB - TEXT TO PHYSICS + DATA CERTIFICATE")
    print("=" * 50)
    while True:
        try:
            text = input("\n> Enter text to seed the physics (or 'q' to quit): ").strip()
            if text.lower() == "q":
                print("Exiting...")
                break
            if not text:
                print("Please enter valid text.")
                continue
            run(text, args)
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="?", help="text that seeds the physical configuration")
    ap.add_argument("--out", default="qnos_text_out")
    ap.add_argument("--h", type=float, default=0.03, help="billiard grid spacing")
    ap.add_argument("--nev", type=int, default=220, help="number of billiard eigenstates")
    ap.add_argument("--shuffles", type=int, default=2000, help="permutation shuffles (multiple of 500)")
    ap.add_argument("--quick", action="store_true", help="coarser grid, fewer states and shuffles")
    ap.add_argument("--show", action="store_true", help="also open the figure window")
    ap.add_argument("--physics", help="path to qnos_physics20262.py if it is not next to this script")
    args = ap.parse_args()
    if args.quick:
        args.h, args.nev, args.shuffles = 0.04, 120, 500
    if args.text:
        run(args.text, args)
    else:
        interactive(args)


if __name__ == "__main__":
    main()
