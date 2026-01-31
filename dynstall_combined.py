# ============================================================
# Combined Dynamic Stall model (Canonical outputs: CL/CD)
# - AeroDyn polar reader (CL/CD)
# - Øye (separation-point lag) -> outputs CL/CD (+ CT/CN derived)
# - Leishman–Beddoes 1st-order (WES-style)
# - IAG = LB + second-order CN correction + drag limiting
# - Optional dynamic inflow coupling wrapper (alpha_eff feedback)
# - Drivers: pitch hysteresis + Tier-B energy/work-per-cycle
# ============================================================

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Tuple
import matplotlib.pyplot as plt

# ============================================================
# Numerics
# ============================================================

EPS = 1e-12
BIG = 1e6


def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)


def safe_div(a, b):
    return a / (b + EPS)


# ============================================================
# Projections between CL/CD and CN/CT
# ============================================================

def cn_from_cl_cd(cl: float, cd: float, alpha_rad: float) -> float:
    return cl * np.cos(alpha_rad) + cd * np.sin(alpha_rad)


def ct_from_cl_cd(cl: float, cd: float, alpha_rad: float) -> float:
    return -cl * np.sin(alpha_rad) + cd * np.cos(alpha_rad)


def cl_cd_from_cn_ct(cn: float, ct: float, alpha_rad: float) -> Tuple[float, float]:
    # Inverse transform:
    # [CN] = [ cos  sin][CL]
    # [CT]   [-sin cos][CD]
    ca = np.cos(alpha_rad)
    sa = np.sin(alpha_rad)
    cl = cn * ca - ct * sa
    cd = cn * sa + ct * ca
    return float(cl), float(cd)


# ============================================================
# Polar interface
# ============================================================

class Polar(Protocol):
    def cl(self, alpha_rad: float) -> float: ...

    def cd(self, alpha_rad: float) -> float: ...

    def cm(self, alpha_rad: float) -> float: ...


@dataclass
class AeroDynPolar:
    """
    Reads AeroDyn-style polar file.
    Expected numeric rows like:
      alpha_deg   Cl    Cd   [Cm]
    Non-numeric rows are ignored.
    """
    alpha_deg: np.ndarray
    cl_data: np.ndarray
    cd_data: np.ndarray
    cm_data: Optional[np.ndarray] = None

    @staticmethod
    def from_file(path: str) -> "AeroDynPolar":
        alpha, cl, cd, cm = [], [], [], []

        with open(path, "r") as f:
            for raw in f:
                s = raw.strip()
                if not s:
                    continue
                parts = s.replace(",", " ").split()
                if len(parts) < 3:
                    continue
                try:
                    vals = [float(p) for p in parts[:4]]
                except ValueError:
                    continue

                alpha.append(vals[0])
                cl.append(vals[1])
                cd.append(vals[2])
                if len(parts) >= 4:
                    cm.append(vals[3])

        if len(alpha) < 10:
            raise ValueError(f"Failed to parse polar table from {path} (rows={len(alpha)}).")

        alpha = np.array(alpha, dtype=float)
        cl = np.array(cl, dtype=float)
        cd = np.array(cd, dtype=float)

        cm_arr = None
        if len(cm) == len(alpha):
            cm_arr = np.array(cm, dtype=float)

        idx = np.argsort(alpha)

        alpha_deg = alpha[idx]
        cl_data = cl[idx]
        cd_data = cd[idx]
        cm_arr = None if cm_arr is None else cm_arr[idx]

        # Extend to [-180, 180] so unsteady models (alpha_f / alpha_eff) won't crash on short polars
        alpha, cl, cd, cm_arr = extend_polar_to_180_viterna(
            alpha, cl, cd, cm_arr,
            AR=None,
            cdmax=None,
            deg_step=1.0,
            cl_scale_highalpha=0.7,
        )

        return AeroDynPolar(alpha_deg=alpha, cl_data=cl, cd_data=cd, cm_data=cm_arr)

    def cl(self, alpha_rad: float) -> float:
        a = np.rad2deg(alpha_rad)
        a = float(np.clip(a, self.alpha_deg[0], self.alpha_deg[-1]))
        return float(np.interp(a, self.alpha_deg, self.cl_data))

    def cd(self, alpha_rad: float) -> float:
        a = np.rad2deg(alpha_rad)
        a = float(np.clip(a, self.alpha_deg[0], self.alpha_deg[-1]))
        return float(np.interp(a, self.alpha_deg, self.cd_data))

    def cm(self, alpha_rad: float) -> float:
        if self.cm_data is None:
            raise ValueError("No CM column in this polar.")
        a = np.rad2deg(alpha_rad)
        a = float(np.clip(a, self.alpha_deg[0], self.alpha_deg[-1]))
        return float(np.interp(a, self.alpha_deg, self.cm_data))


# ============================================================
# Polar extrapolation helper: extend CL/CD(/CM) to [-180, 180] deg
# - Prevents alpha_f or alpha_eff from blowing up when polar tables are short
# - Uses Viterna-style high-alpha extension up to 90 deg and reflection beyond
# ============================================================

def _viterna_params(alpha_s_rad: float, cl_s: float, cd_s: float, cdmax: float) -> Tuple[float, float]:
    sa = np.sin(alpha_s_rad)
    ca = np.cos(alpha_s_rad)
    ca2 = ca * ca
    if ca2 < 1e-10:
        raise ValueError("alpha_s too close to 90deg; cannot form Viterna parameters safely.")
    B = (cd_s - cdmax * (sa * sa)) / ca2
    A = (cl_s - 0.5 * cdmax * np.sin(2.0 * alpha_s_rad)) * sa / ca2
    return float(A), float(B)


def _viterna_eval(alpha_rad: float, A: float, B: float, cdmax: float) -> Tuple[float, float]:
    sa = np.sin(alpha_rad)
    ca = np.cos(alpha_rad)
    cd = cdmax * (sa * sa) + B * (ca * ca)
    if abs(sa) < 1e-10:
        cl = 0.5 * cdmax * np.sin(2.0 * alpha_rad)
    else:
        cl = 0.5 * cdmax * np.sin(2.0 * alpha_rad) + A * (ca * ca) / sa
    return float(cl), float(max(cd, 0.0))


def extend_polar_to_180_viterna(
        alpha_deg: np.ndarray,
        cl: np.ndarray,
        cd: np.ndarray,
        cm: Optional[np.ndarray] = None,
        *,
        AR: Optional[float] = None,
        cdmax: Optional[float] = None,
        deg_step: float = 1.0,
        cl_scale_highalpha: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    alpha_deg = np.asarray(alpha_deg, dtype=float)
    cl = np.asarray(cl, dtype=float)
    cd = np.asarray(cd, dtype=float)
    cm_arr = None if cm is None else np.asarray(cm, dtype=float)

    idx = np.argsort(alpha_deg)
    alpha_deg = alpha_deg[idx]
    cl = cl[idx]
    cd = cd[idx]
    if cm_arr is not None:
        cm_arr = cm_arr[idx]

    if float(alpha_deg[0]) <= -170.0 and float(alpha_deg[-1]) >= 170.0:
        return alpha_deg, cl, cd, cm_arr

    if cdmax is None:
        if AR is not None:
            cdmax = 1.11 + 0.018 * float(AR)
        else:
            cdmax = max(1.8, float(np.nanmax(cd)))

    a_s = float(np.max(alpha_deg))
    a_s = min(a_s, 89.0)
    cl_s = float(np.interp(a_s, alpha_deg, cl))
    cd_s = float(np.interp(a_s, alpha_deg, cd))
    A, B = _viterna_params(np.deg2rad(a_s), cl_s, cd_s, float(cdmax))

    def clcd_pos(a_pos_deg: float) -> Tuple[float, float]:
        if a_pos_deg <= a_s:
            return float(np.interp(a_pos_deg, alpha_deg, cl)), float(np.interp(a_pos_deg, alpha_deg, cd))
        cl_v, cd_v = _viterna_eval(np.deg2rad(a_pos_deg), A, B, float(cdmax))
        return float(cl_scale_highalpha * cl_v), float(cd_v)

    a_ext = np.arange(-180.0, 180.0 + deg_step, deg_step)
    cl_ext = np.zeros_like(a_ext)
    cd_ext = np.zeros_like(a_ext)
    cm_ext = None if cm_arr is None else np.zeros_like(a_ext)

    a_min, a_max = float(alpha_deg[0]), float(alpha_deg[-1])

    for i, a in enumerate(a_ext):
        if a_min <= a <= a_max:
            cl_ext[i] = float(np.interp(a, alpha_deg, cl))
            cd_ext[i] = float(np.interp(a, alpha_deg, cd))
            if cm_ext is not None:
                cm_ext[i] = float(np.interp(a, alpha_deg, cm_arr))
            continue

        sgn = 1.0 if a >= 0.0 else -1.0
        aa = abs(a)

        if aa <= 90.0:
            cl_p, cd_p = clcd_pos(aa)
            cl_ext[i] = sgn * cl_p
            cd_ext[i] = cd_p
        else:
            mirror = 180.0 - aa
            cl_p, cd_p = clcd_pos(mirror)
            cl_ext[i] = -sgn * cl_p
            cd_ext[i] = cd_p

        if cm_ext is not None:
            cm_ext[i] = float(np.interp(np.clip(a, a_min, a_max), alpha_deg, cm_arr))

    return a_ext, cl_ext, cd_ext, cm_ext


# ============================================================
# Polar analysis helpers (slope, alpha0, stall estimate)
# ============================================================

def estimate_alpha0_linear(polar: AeroDynPolar, window_deg: float = 8.0) -> float:
    a = polar.alpha_deg
    cl = polar.cl_data
    m = (a >= -window_deg) & (a <= window_deg)
    if np.sum(m) < 4:
        # fallback: use broader window
        m = (a >= -40.0) & (a <= 40.0)

    a_fit = a[m]
    cl_fit = cl[m]
    A = np.vstack([a_fit, np.ones_like(a_fit)]).T
    m_slope, b0 = np.linalg.lstsq(A, cl_fit, rcond=None)[0]
    if abs(m_slope) < 1e-12:
        return float("nan")
    return float(-b0 / m_slope)


def estimate_cl_alpha(polar: AeroDynPolar, window_deg: float = 8.0) -> float:
    a = polar.alpha_deg
    cl = polar.cl_data
    m = (a >= -window_deg) & (a <= window_deg)
    if np.sum(m) < 4:
        m = (a >= -40.0) & (a <= 40.0)
    a_fit = np.deg2rad(a[m])
    cl_fit = cl[m]
    A = np.vstack([a_fit, np.ones_like(a_fit)]).T
    m_slope, _ = np.linalg.lstsq(A, cl_fit, rcond=None)[0]
    return float(m_slope)


def estimate_alpha_ss_from_cl(polar: AeroDynPolar, core_lim_deg: float = 40.0) -> float:
    """
    Engineering estimate of static stall angle magnitude alpha_ss from |CL| peak
    in a limited "core" range to avoid extended-table endcaps.
    """
    a = polar.alpha_deg
    cl = polar.cl_data
    m = (a >= -core_lim_deg) & (a <= core_lim_deg)
    if np.sum(m) < 10:
        m = np.ones_like(a, dtype=bool)
    a_core = a[m]
    cl_core = cl[m]
    i = int(np.argmax(np.abs(cl_core)))
    return float(abs(a_core[i]))


def dcn_dalpha_est(polar: Polar, alpha0_rad: float = 0.0, h: float = 1e-4) -> float:
    cl_p = polar.cl(alpha0_rad + h)
    cd_p = polar.cd(alpha0_rad + h)
    cn_p = cn_from_cl_cd(cl_p, cd_p, alpha0_rad + h)

    cl_m = polar.cl(alpha0_rad - h)
    cd_m = polar.cd(alpha0_rad - h)
    cn_m = cn_from_cl_cd(cl_m, cd_m, alpha0_rad - h)

    return float((cn_p - cn_m) / (2 * h))


# ============================================================
# Dynamic Stall model interface (canonical outputs CL/CD)
# ============================================================

class DynamicStallModel(Protocol):
    def reset(self) -> None: ...

    def step(
            self,
            alpha_rad: float,
            alpha_dot_rad: float,
            U: float,
            dt: float,
            polar: Polar
    ) -> Dict[str, float]: ...


@dataclass
class NoDynamicStall:
    def reset(self) -> None:
        pass

    def step(self, alpha_rad: float, alpha_dot_rad: float, U: float, dt: float, polar: Polar) -> Dict[str, float]:
        cl = polar.cl(alpha_rad)
        cd = polar.cd(alpha_rad)
        ct = ct_from_cl_cd(cl, cd, alpha_rad)
        cn = cn_from_cl_cd(cl, cd, alpha_rad)
        # Static CM from polar
        try:
            cm = polar.cm(alpha_rad)
        except Exception:
            cm = 0.0
        return {"cl": cl, "cd": cd, "ct": ct, "cn": cn, "cm": cm}


# ============================================================
# Øye Dynamic Stall (CL-based separation point lag)
# ============================================================

@dataclass
class OyeDynamicStallCL:
    """
    Øye dynamic stall (separation-point lag) with canonical CL/CD output.
    Reconstruct:
      CL_dyn = f_s * CL_inv + (1-f_s) * CL_fs

    CD: either static CD, or optional Bergami-style adjustment.
    """
    c: float
    Tf0: float = 6.0
    alpha_fit_window_deg: float = 8.0
    stall_margin: float = 1.0
    use_bergami_cd: bool = True

    # cached polar-derived
    cl_alpha: Optional[float] = None
    alpha0_rad: Optional[float] = None
    alpha_ss_deg: Optional[float] = None

    # state
    f_s: float = 1.0

    def reset(self) -> None:
        self.f_s = 1.0

    def _ensure_cached(self, polar: Polar) -> None:
        if self.cl_alpha is not None and self.alpha0_rad is not None and self.alpha_ss_deg is not None:
            return

        if isinstance(polar, AeroDynPolar):
            self.cl_alpha = estimate_cl_alpha(polar, self.alpha_fit_window_deg)
            a0_deg = estimate_alpha0_linear(polar, self.alpha_fit_window_deg)
            self.alpha0_rad = float(np.deg2rad(a0_deg))
            self.alpha_ss_deg = estimate_alpha_ss_from_cl(polar, 40.0)
        else:
            # reasonable defaults
            self.cl_alpha = 2.0 * np.pi
            self.alpha0_rad = 0.0
            self.alpha_ss_deg = 15.0

    @staticmethod
    def _cl_fully_separated(alpha_rad: float) -> float:
        # Kirchhoff-like fully separated lift model
        return float(2.0 * np.sin(alpha_rad) * np.cos(alpha_rad))

    def step(self, alpha_rad: float, alpha_dot_rad: float, U: float, dt: float, polar: Polar) -> Dict[str, float]:
        self._ensure_cached(polar)

        Ueff = max(U, 0.1)

        cl_alpha = float(self.cl_alpha)
        alpha0 = float(self.alpha0_rad)

        # static from polar
        cl_st = float(polar.cl(alpha_rad))
        cd_st = float(polar.cd(alpha_rad))

        # inviscid extrapolation near linear region
        cl_inv = float(cl_alpha * (alpha_rad - alpha0))

        # fully separated
        cl_fs = self._cl_fully_separated(alpha_rad)

        # steady separation function f_st derived from Kirchhoff relation
        denom = max(abs(cl_inv), 1e-9)
        ratio = max(cl_st / denom, 0.0)
        f_st = (2.0 * np.sqrt(ratio) - 1.0) ** 2
        f_st = float(np.clip(f_st, 0.0, 1.0))

        # lag time constant Tf = Tf0 * Tu with Tu = c/(2U)
        # We do not have c here directly; we treat Tf0 as "effective" in seconds via user’s dt/U scaling.
        # Better: pass chord c in constructor if you want exact Tu. Here: approximate Tu with 1/(2U).
        Tu = self.c / (2.0 * Ueff)
        Tf = max(self.Tf0 * Tu, 1e-6)

        a = np.exp(-dt / Tf)
        self.f_s = float(a * self.f_s + (1.0 - a) * f_st)

        cl_dyn = float(self.f_s * cl_inv + (1.0 - self.f_s) * cl_fs)

        if self.use_bergami_cd:
            # Bergami-style drag adjustment needs Cd0 at alpha0
            cd0 = float(polar.cd(alpha0))
            term = 0.5 * (np.sqrt(max(f_st, 0.0)) - np.sqrt(max(self.f_s, 0.0))) - 0.25 * (self.f_s - f_st)
            cd_dyn = float(cd_st + (cd_st - cd0) * term)
            cd_dyn = max(cd_dyn, 0.0)
        else:
            cd_dyn = cd_st

        ct_dyn = ct_from_cl_cd(cl_dyn, cd_dyn, alpha_rad)
        cn_dyn = cn_from_cl_cd(cl_dyn, cd_dyn, alpha_rad)

        # Dynamic CM: use Kirchhoff-style center-of-pressure shift
        try:
            cm_st = polar.cm(alpha_rad)
        except Exception:
            cm_st = 0.0
        # Center of pressure moves aft as flow separates (f_s -> 0)
        # Use smaller coefficient for realistic behavior
        delta_xcp = 0.12 * (1.0 - np.sqrt(max(self.f_s, 0.0)))
        cm_dyn = float(cm_st - delta_xcp * cn_dyn)

        return {
            "cl": cl_dyn,
            "cd": cd_dyn,
            "ct": float(ct_dyn),
            "cn": float(cn_dyn),
            "cm": cm_dyn,
            "f_s": float(self.f_s),
            "f_st": float(f_st),
            "cl_inv": float(cl_inv),
            "cl_fs": float(cl_fs),
        }


# ============================================================
# Leishman–Beddoes First Order (WES-style)
# ============================================================

@dataclass
class LeishmanBeddoesFirstOrder:
    """
    First-order LB indicial implementation (attached + separated + vortex),
    returns CL/CD plus useful internals (CN, CT, alpha_f, f2, etc.)
    """
    c: float

    # Indicial constants (typical LB values)
    A1: float = 0.3
    A2: float = 0.7
    b1: float = 0.14
    b2: float = 0.53
    Kalpha: float = 0.75
    beta: float = 1.0  # incompressible

    # Lags in s-domain (s = 2 U t / c)
    Tp: float = 1.7
    Tf: float = 3.0
    Tv: float = 6.0

    # Vortex staging
    Tvl: float = 2.0
    alpha_crit_rad: float = np.deg2rad(15.0)

    # states
    X: float = 0.0
    Y: float = 0.0
    D: float = 0.0
    Dp: float = 0.0
    Df: float = 0.0
    tau_v: float = 0.0
    CNv: float = 0.0
    Cv_prev: float = 0.0
    fn_prev: float = 1.0

    prev_alpha: Optional[float] = None
    prev_dalpha: float = 0.0
    prev_CPN: float = 0.0

    # cached polar-derived constants
    _dCN_dalpha: Optional[float] = None
    _alpha0_inv: Optional[float] = None
    _alpha0_visc: Optional[float] = None

    def reset(self) -> None:
        self.X = self.Y = 0.0
        self.D = self.Dp = self.Df = 0.0
        self.tau_v = 0.0
        self.CNv = 0.0
        self.Cv_prev = 0.0
        self.fn_prev = 1.0
        self.prev_alpha = None
        self.prev_dalpha = 0.0
        self.prev_CPN = 0.0
        self._dCN_dalpha = None
        self._alpha0_inv = None
        self._alpha0_visc = None

    def _init_from_polar(self, polar: Polar) -> None:
        self._dCN_dalpha = float(dcn_dalpha_est(polar, 0.0))

        if isinstance(polar, AeroDynPolar):
            a0_deg = estimate_alpha0_linear(polar, window_deg=8.0)
            self._alpha0_visc = float(np.deg2rad(a0_deg))
        else:
            self._alpha0_visc = 0.0

        self._alpha0_inv = float(self._alpha0_visc)

    def step(self, alpha_rad: float, alpha_dot_rad: float, U: float, dt: float, polar: Polar) -> Dict[str, float]:
        if self._dCN_dalpha is None:
            self._init_from_polar(polar)

        Ueff = max(U, 0.1)

        dCN = float(self._dCN_dalpha)
        alpha0_inv = float(self._alpha0_inv)
        alpha0_visc = float(self._alpha0_visc)

        # nondimensional timestep in s = 2 U t / c
        ds = 2.0 * Ueff * dt / max(self.c, 1e-12)

        if self.prev_alpha is None:
            dalpha = 0.0
        else:
            dalpha = alpha_rad - self.prev_alpha

        # 1) Attached flow indicial (X,Y)
        e1 = np.exp(-self.b1 * (self.beta ** 2) * ds)
        e2 = np.exp(-self.b2 * (self.beta ** 2) * ds)
        self.X = self.X * e1 + self.A1 * dalpha * np.exp(-0.5 * self.b1 * (self.beta ** 2) * ds)
        self.Y = self.Y * e2 + self.A2 * dalpha * np.exp(-0.5 * self.b2 * (self.beta ** 2) * ds)

        alpha_e = alpha_rad - self.X - self.Y
        CNc = dCN * (alpha_e - alpha0_inv)

        # 2) Impulsive term (deficiency)
        tau_imp = max(self.Kalpha * (self.c / Ueff), 1e-9)
        aD = np.exp(-dt / tau_imp)

        # stable alpha_ddot estimate
        alpha_ddot = (alpha_dot_rad - self.prev_dalpha) / max(dt, 1e-12)

        # first-order lag on alpha_ddot (deficiency)
        self.D = aD * self.D + (1.0 - aD) * alpha_ddot

        # impulsive CN term
        CNi = 4.0 * self.Kalpha * (self.c / Ueff) * (alpha_dot_rad - self.D)

        CPN = CNc + CNi

        # 3) Pressure lag -> alpha_f
        aP = np.exp(-ds / max(self.Tp, 1e-9))
        self.Dp = self.Dp * aP + (CPN - self.prev_CPN) * np.exp(-0.5 * ds / max(self.Tp, 1e-9))
        CPN1 = CPN - self.Dp
        alpha_f = alpha_rad if abs(dCN) < 1e-12 else (alpha0_inv + CPN1 / dCN)

        # 4) Separation inversion fn(alpha_f) and lag to f2
        cl_f = float(polar.cl(alpha_f))
        cd_f = float(polar.cd(alpha_f))
        CN_visc_f = cn_from_cl_cd(cl_f, cd_f, alpha_f)

        denom = dCN * (alpha_f - alpha0_visc)
        if abs(denom) < 1e-12:
            fn = 1.0
        else:
            r = max(CN_visc_f / denom, 0.0)
            fn = (2.0 * np.sqrt(r) - 1.0) ** 2
        fn = float(np.clip(fn, 0.0, 1.0))

        aF = np.exp(-ds / max(self.Tf, 1e-9))
        self.Df = self.Df * aF + (fn - self.fn_prev) * np.exp(-0.5 * ds / max(self.Tf, 1e-9))
        f2 = float(np.clip(fn - self.Df, 0.0, 1.0))
        self.fn_prev = fn

        # --- Separated (viscous) CN: use polar at alpha_f to prevent runaway ---
        CN_visc_af = cn_from_cl_cd(cl_f, cd_f, alpha_f)

        # Blend: when f2=1 -> mostly attached CNc, when f2=0 -> polar viscous CN at alpha_f
        CNf = f2 * CNc + (1.0 - f2) * CN_visc_af

        # 5) Vortex lift module
        cl_crit = float(polar.cl(self.alpha_crit_rad))
        cd_crit = float(polar.cd(self.alpha_crit_rad))
        CN_crit = float(cn_from_cl_cd(cl_crit, cd_crit, self.alpha_crit_rad))

        ds_v = ds

        if CPN1 > CN_crit:
            self.tau_v = self.tau_v + 0.33 * ds_v
        elif (CPN1 < CN_crit) and (dalpha >= 0.0):
            self.tau_v = 0.0

        Cv = CNc - CNf
        aV = np.exp(-ds / max(self.Tv, 1e-9))
        if 0.0 < self.tau_v < self.Tvl:
            self.CNv = self.CNv * aV + (Cv - self.Cv_prev) * np.exp(-0.5 * ds / max(self.Tv, 1e-9))
        else:
            self.CNv = self.CNv * aV
        self.Cv_prev = Cv

        CND1 = CNf + self.CNv

        # Tangential force: use static CT at alpha_f
        CTf = ct_from_cl_cd(cl_f, cd_f, alpha_f)

        # Convert to CL/CD at geometric alpha
        CL1 = CND1 * np.cos(alpha_rad) - CTf * np.sin(alpha_rad)
        CD1 = CND1 * np.sin(alpha_rad) + CTf * np.cos(alpha_rad)

        # update history
        self.prev_alpha = alpha_rad
        self.prev_dalpha = alpha_dot_rad
        self.prev_CPN = CPN

        # Dynamic CM: Leishman-Beddoes moment model
        # CM = CM_static + CM_f (separation) + CM_v (vortex)
        try:
            cm_st = polar.cm(alpha_rad)
        except Exception:
            cm_st = 0.0

        # Separation-induced moment: center of pressure moves aft
        # Use smaller coefficients for more realistic behavior
        # delta_xcp ~ K1*(1-sqrt(f2)) where f2=1 is attached, f2=0 is separated
        K1 = 0.10  # reduced from 0.25 - max c.p. shift ~10% chord
        delta_xcp = K1 * (1.0 - np.sqrt(max(f2, 0.0)))
        CM_f = -delta_xcp * CND1

        # Vortex-induced moment: standard LEV (leading-edge vortex) physics
        # Vortex forms at trailing edge (xcp=0.25) and convects forward over Tvl,
        # settling near separation point (~0.10). Position decays with vortex age.
        xcp_v = 0.25 - 0.15 * min(self.tau_v / max(self.Tvl, 1e-6), 1.0)
        CM_v = -xcp_v * self.CNv

        cm_dyn = float(cm_st + CM_f + CM_v)

        return {
            "cl": float(CL1),
            "cd": float(max(CD1, 0.0)),
            "ct": float(ct_from_cl_cd(CL1, CD1, alpha_rad)),
            "cn": float(cn_from_cl_cd(CL1, CD1, alpha_rad)),
            "cm": cm_dyn,
            # internals
            "cn1": float(CND1),
            "ct1": float(CTf),
            "alpha_f": float(alpha_f),
            "fn": float(fn),
            "f2": float(f2),
            "cpn1": float(CPN1),
            "cnv": float(self.CNv),
            "tau_v": float(self.tau_v),
            "cnc": float(CNc),
            "cni": float(CNi),
            "cnf": float(CNf),
        }


# ============================================================
# IAG Second-Order Correction on CN (WES Eq 73-76 style)
# ============================================================

@dataclass
class IAGSecondOrderCN:
    c: float
    ks: float = 0.2
    alpha_crit_rad: float = np.deg2rad(15.0)

    z: float = 0.0  # d2CN state
    v: float = 0.0  # derivative wrt s
    prev_alpha: Optional[float] = None

    _dCN_dalpha: Optional[float] = None
    _alpha0_inv: float = 0.0

    def reset(self) -> None:
        self.z = 0.0
        self.v = 0.0
        self.prev_alpha = None
        self.prev_dCINV = None
        self._dCN_dalpha = None
        self._alpha0_inv = 0.0

    def _init_from_polar(self, polar: Polar) -> None:
        self._dCN_dalpha = float(dcn_dalpha_est(polar, 0.0))
        if isinstance(polar, AeroDynPolar):
            a0_deg = estimate_alpha0_linear(polar, window_deg=8.0)
            self._alpha0_inv = float(np.deg2rad(a0_deg))
        else:
            self._alpha0_inv = 0.0

    def step(self, alpha_rad: float, U: float, dt: float, polar: Polar) -> Dict[str, float]:
        if self._dCN_dalpha is None:
            self._init_from_polar(polar)

        Ueff = max(U, 0.1)
        dCN = float(self._dCN_dalpha)
        alpha0_inv = float(self._alpha0_inv)

        ds = 2.0 * Ueff * dt / max(self.c, 1e-12)

        if self.prev_alpha is None:
            alpha_dot_t = 0.0
        else:
            alpha_dot_t = (alpha_rad - self.prev_alpha) / max(dt, 1e-12)

        tau = self.c / (2.0 * Ueff)
        alpha_dot_s = tau * alpha_dot_t

        # inviscid CN vs viscous CN
        CN_inv = dCN * (alpha_rad - alpha0_inv)
        cl = polar.cl(alpha_rad)
        cd = polar.cd(alpha_rad)
        CN_visc = cn_from_cl_cd(cl, cd, alpha_rad)
        dCINV = CN_inv - CN_visc
        if getattr(self, "prev_dCINV", None) is None:
            CdotINV_s = 0.0
        else:
            CdotINV_s = (dCINV - self.prev_dCINV) / max(ds, 1e-12)

        self.prev_dCINV = dCINV
        # forcing (Eq 76 style)
        F2 = 0.5 * self.ks * (-0.15 * dCINV + 0.05 * CdotINV_s)

        # Kf20, Kf21
        Kf20 = 20.0 * (self.ks ** 2) * (1.0 + 3.0 * (self.z ** 2)) * (1.0 + 3.0 * (alpha_dot_s ** 2))

        if alpha_dot_s > 0.0:
            Kf21 = 150.0 * self.ks * (-0.01 * (dCINV - 0.5) + 2.0 * (self.z ** 2))
        else:
            if alpha_rad >= self.alpha_crit_rad:
                Kf21 = 30.0 * self.ks * (-0.01 * (dCINV - 0.5) + 14.0 * (self.z ** 2))
            else:
                Kf21 = 0.2 * self.ks

        # integrate in s-domain
        dv = (F2 - Kf21 * self.v - Kf20 * self.z)
        self.v += ds * dv
        self.z += ds * self.v

        self.prev_alpha = alpha_rad
        return {"d2cn": float(self.z), "d2cn_dot": float(self.v), "F2": float(F2), "Kf20": float(Kf20),
                "Kf21": float(Kf21)}


# ============================================================
# Full IAG Dynamic Stall (LB + 2nd-order CN + drag limiting)
# ============================================================

@dataclass
class IAGDynamicStallCL:
    c: float
    ks: float = 0.2
    alpha_crit_rad: float = np.deg2rad(15.0)

    # drag limiting
    zeta_v: float = 0.76

    # LB params
    Tp: float = 1.7
    Tf: float = 3.0
    Tv: float = 6.0
    Tvl: float = 2.0
    Kalpha: float = 0.75
    A1: float = 0.3
    A2: float = 0.7
    b1: float = 0.14
    b2: float = 0.53

    _lb: Optional[LeishmanBeddoesFirstOrder] = None
    _d2: Optional[IAGSecondOrderCN] = None
    _dCN_dalpha: Optional[float] = None
    prev_CPN: float = 0.0

    def reset(self) -> None:
        self._lb = LeishmanBeddoesFirstOrder(
            c=self.c,
            A1=self.A1, A2=self.A2, b1=self.b1, b2=self.b2,
            Kalpha=self.Kalpha, Tp=self.Tp, Tf=self.Tf, Tv=self.Tv,
            Tvl=self.Tvl, alpha_crit_rad=self.alpha_crit_rad
        )
        self._d2 = IAGSecondOrderCN(c=self.c, ks=self.ks, alpha_crit_rad=self.alpha_crit_rad)
        self._lb.reset()
        self._d2.reset()
        self._dCN_dalpha = None
        self.prev_CPN = 0.0

    def step(self, alpha_rad: float, alpha_dot_rad: float, U: float, dt: float, polar: Polar) -> Dict[str, float]:
        if self._lb is None or self._d2 is None:
            self.reset()

        if self._dCN_dalpha is None:
            self._dCN_dalpha = float(dcn_dalpha_est(polar, 0.0))
        dCN = float(self._dCN_dalpha)

        # First-order LB
        o1 = self._lb.step(alpha_rad, alpha_dot_rad, U, dt, polar)
        CN1 = float(o1["cn"])

        o2 = self._d2.step(alpha_rad, U, dt, polar)
        d2CN = float(o2["d2cn"])

        # --------------------------------------------------
        # Change 2: gate + clamp the 2nd-order correction
        # --------------------------------------------------

        # Use LB separation indicator: f2 ~ 1 attached, f2 ~ 0 separated
        f2 = float(o1.get("f2", 1.0))

        # Gate: apply D2 mainly in/near stall on UPSTROKE
        gate = 1.0 if (alpha_rad > self.alpha_crit_rad - np.deg2rad(2.0) and alpha_dot_rad > 0.0) else 0.0

        # Optionally also scale by separation (more separated -> more correction)
        sep_scale = 0.2 + 0.8 * (1.0 - f2)  # never fully off
        d2CN_eff = d2CN * gate * sep_scale

        CN_pol = cn_from_cl_cd(polar.cl(alpha_rad), polar.cd(alpha_rad), alpha_rad)

        # Clamp: don't let D2 dominate the base CN1
        limit = 0.3 * max(1.0, abs(CN_pol))
        d2CN_eff = float(np.clip(d2CN_eff, -limit, limit))

        CND = CN1 + d2CN_eff

        # --------------------------------------------------
        # Change 2b: keep tangential force consistent by using LB's CT
        # (IAG adds a CN correction; CT is taken from the base LB viscous model)

        # --------------------------------------------------
        # Use LB tangential force (already consistent with the LB unsteady model)
        CTD = float(o1.get("ct", 0.0))  # this is CT in the geometric frame (alpha_rad)
        CL = CND * np.cos(alpha_rad) - CTD * np.sin(alpha_rad)
        CD = CND * np.sin(alpha_rad) + CTD * np.cos(alpha_rad)

        # Drag limiting (WES Eq 63–64 style)
        f2 = float(o1.get("f2", 1.0))
        zeta = (1.0 / np.pi) * dCN * (1.0 + np.sqrt(f2) / 2.0) ** 2
        CD_static = float(polar.cd(alpha_rad))

        CPN_now = float(o1.get("cpn1", 0.0))
        dCPN = CPN_now - self.prev_CPN

        if zeta >= self.zeta_v:
            if (CD > 1.2 * CD_static) and (dCPN >= 0.0):
                CD = 1.2 * CD_static
            elif dCPN < 0.0:
                CD = CD_static

        self.prev_CPN = CPN_now
        CD_lb = float(o1["cd"])
        CD = max(CD, CD_lb, CD_static)

        CD = float(max(CD, 0.0))

        CT = float(ct_from_cl_cd(CL, CD, alpha_rad))
        CN = float(cn_from_cl_cd(CL, CD, alpha_rad))

        # Compute IAG CM: same physics as LB (separation + vortex c.p. shift)
        # but using IAG's CN values
        try:
            cm_st = polar.cm(alpha_rad)
        except Exception:
            cm_st = 0.0

        # Separation-induced moment: c.p. moves aft as flow separates
        # Same physical basis as LB: delta_xcp ~ K1*(1-sqrt(f2))
        K1 = 0.10  # max c.p. shift ~10% chord at full separation
        delta_xcp = K1 * (1.0 - np.sqrt(max(f2, 0.0)))
        CM_f = -delta_xcp * CND

        # Vortex-induced moment from underlying LB
        # Use same standard LEV physics as base LB model
        CNv = float(o1.get("cnv", 0.0))
        tau_v = float(o1.get("tau_v", 0.0))
        xcp_v = 0.25 - 0.15 * min(tau_v / max(self.Tvl, 1e-6), 1.0)
        CM_v = -xcp_v * CNv

        cm_dyn = float(cm_st + CM_f + CM_v)

        out = {
            "cl": float(CL),
            "cd": float(CD),
            "ct": CT,
            "cn": CN,
            "cm": cm_dyn,
            "cn1": CN1,
            "d2cn_raw": float(d2CN),
            "d2cn_eff": float(d2CN_eff),
            "zeta": float(zeta),
        }
        # carry internals with prefixes
        for k, v in o1.items():
            if k in ("cl", "cd", "ct", "cn", "cm"):
                continue
            if isinstance(v, (float, int, np.floating, np.integer)):
                out[f"lb_{k}"] = float(v)
        for k, v in o2.items():
            if isinstance(v, (float, int, np.floating, np.integer)):
                out[f"d2_{k}"] = float(v)

        return out


# ============================================================
# Dynamic inflow model (optional coupling wrapper)
# ============================================================

@dataclass
class DynamicInflowModel:
    tau_factor: float = 1.0
    gain: float = 0.05
    w: float = 0.0

    def reset(self) -> None:
        self.w = 0.0

    def step(self, CN: float, U_inf: float, R: float, dt: float) -> float:
        U = max(U_inf, 0.1)
        w_qs = self.gain * CN * U
        tau = self.tau_factor * R / (2 * U)
        self.w += dt * (w_qs - self.w) / max(tau, 1e-6)
        return self.w


@dataclass
class CoupledInflowWrapper:
    """
    Wrap any DynamicStallModel and apply induced velocity feedback:
      alpha_eff = alpha_geo - atan(w/U)
    """
    ds: DynamicStallModel
    inflow: DynamicInflowModel
    R: float

    def reset(self) -> None:
        self.ds.reset()
        self.inflow.reset()

    def step(self, alpha_geo_rad: float, alpha_dot_geo_rad: float, U_inf: float, dt: float, polar: Polar) -> Dict[
        str, float]:
        w = self.inflow.w
        alpha_eff = alpha_geo_rad - np.arctan2(w, max(U_inf, 0.1))

        out = self.ds.step(alpha_eff, alpha_dot_geo_rad, U_inf, dt, polar)

        # update inflow using CN from outputs
        CN = float(out.get("cn", 0.0))
        self.inflow.step(CN=CN, U_inf=U_inf, R=self.R, dt=dt)

        out["alpha_eff_rad"] = float(alpha_eff)
        out["w"] = float(self.inflow.w)
        return out


# ============================================================
# Driver 1: Pitch sinusoid (hysteresis)
# ============================================================

def run_pitch_sine(
        model: DynamicStallModel,
        polar: Polar,
        dt: float = 1e-3,
        t_end: float = 2.0,
        alpha_mean_deg: float = 2.0,
        alpha_amp_deg: float = 12.0,
        freq_hz: float = 1.0,
        U: float = 20.0,
) -> Dict[str, np.ndarray]:
    t = np.arange(0.0, t_end, dt)

    alpha_deg = alpha_mean_deg + alpha_amp_deg * np.sin(2 * np.pi * freq_hz * t)
    alpha_dot_deg = 2 * np.pi * freq_hz * alpha_amp_deg * np.cos(2 * np.pi * freq_hz * t)

    alpha = np.deg2rad(alpha_deg)
    alpha_dot = np.deg2rad(alpha_dot_deg)

    model.reset()

    cl = np.zeros_like(t)
    cd = np.zeros_like(t)
    cn = np.zeros_like(t)
    ct = np.zeros_like(t)
    cm = np.zeros_like(t)

    for i in range(len(t)):
        out = model.step(alpha[i], alpha_dot[i], U, dt, polar)

        cl[i] = out["cl"]
        cd[i] = out["cd"]
        cm[i] = out.get("cm", 0.0)

        # ALWAYS recompute projections consistently
        ct[i] = ct_from_cl_cd(cl[i], cd[i], alpha[i])
        cn[i] = cn_from_cl_cd(cl[i], cd[i], alpha[i])

    return {
        "t": t,
        "alpha_rad": alpha,
        "alpha_deg": alpha_deg,
        "alpha_dot_rad": alpha_dot,
        "cl": cl,
        "cd": cd,
        "cn": cn,
        "ct": ct,
        "cm": cm,
    }


# ============================================================
# Driver 2: Tier-B prescribed motion (energy/work-per-cycle)
# ============================================================

@dataclass
class TierBConfig:
    rho: float = 1.225
    U: float = 30.0
    c: float = 1.0
    b: float = 1.0
    alpha0_deg: float = 8.0
    X: float = 0.2
    f_hz: float = 1.0
    dt: float = 1e-3
    n_cycles: int = 20
    discard_cycles: int = 5


def run_tierB(cfg: TierBConfig, polar: Polar, ds: DynamicStallModel) -> Dict[str, np.ndarray]:
    ds.reset()
    w = 2 * np.pi * cfg.f_hz
    T = 1.0 / cfg.f_hz
    n_total = int(cfg.n_cycles * T / cfg.dt)
    t = np.arange(n_total) * cfg.dt

    x = cfg.X * np.sin(w * t)
    xdot = cfg.X * w * np.cos(w * t)

    alpha0 = np.deg2rad(cfg.alpha0_deg)
    alpha = alpha0 + (xdot / cfg.U)
    alpha_dot = np.gradient(alpha, cfg.dt)

    cl = np.zeros_like(t)
    cd = np.zeros_like(t)
    ct = np.zeros_like(t)
    cn = np.zeros_like(t)

    Fa = np.zeros_like(t)
    power = np.zeros_like(t)

    d2raw = np.full_like(t, np.nan, dtype=float)

    q = 0.5 * cfg.rho * cfg.U ** 2

    for i in range(n_total):
        out = ds.step(alpha[i], alpha_dot[i], cfg.U, cfg.dt, polar)
        if "d2cn_raw" in out:
            d2raw[i] = out["d2cn_raw"]

        cl[i] = out["cl"]
        cd[i] = out["cd"]

        # FORCE consistent projections
        ct[i] = ct_from_cl_cd(cl[i], cd[i], alpha[i])
        cn[i] = cn_from_cl_cd(cl[i], cd[i], alpha[i])

        Fa[i] = q * cfg.c * cfg.b * ct[i]
        power[i] = Fa[i] * xdot[i]

    i0 = int(cfg.discard_cycles * T / cfg.dt)
    t_ss = t[i0:]
    power_ss = power[i0:]

    samples_per_cycle = int(T / cfg.dt)
    n_cycles_ss = len(t_ss) // samples_per_cycle

    W = []
    for k in range(n_cycles_ss):
        j0 = k * samples_per_cycle
        j1 = (k + 1) * samples_per_cycle
        Wk = np.trapezoid(power_ss[j0:j1], t_ss[j0:j1])
        W.append(Wk)
    W = np.array(W, dtype=float)

    return {
        "t": t,
        "x": x,
        "xdot": xdot,
        "alpha_rad": alpha,
        "cl": cl,
        "cd": cd,
        "ct": ct,
        "cn": cn,
        "Fa": Fa,
        "power": power,
        "W_per_cycle": W,
        "W_mean": float(np.mean(W)) if len(W) else np.nan,
        "W_std": float(np.std(W)) if len(W) else np.nan,
        "T": float(T),
        "reduced_frequency_k": float(np.pi * cfg.f_hz * cfg.c / cfg.U),
        "d2cn_raw": d2raw,

    }


# ============================================================
# Plot helpers
# ============================================================

def plot_hysteresis(alpha_deg: np.ndarray, y: np.ndarray, ylabel: str, title: str):
    plt.figure(figsize=(7, 6))
    plt.plot(alpha_deg, y, lw=2)
    plt.xlabel(r"$\alpha$ (deg)")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_time_series(t: np.ndarray, series: Dict[str, np.ndarray], title: str):
    plt.figure(figsize=(9, 5))
    for k, v in series.items():
        plt.plot(t, v, lw=1.8, label=k)
    plt.xlabel("Time (s)")
    plt.grid(True)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ============================================================
# Main demo
# ============================================================

if __name__ == "__main__":
    polar = AeroDynPolar.from_file("OSU_S809_static_LEGR_1000K.dat")
    print("Polar sanity:")
    print("CL(0)=", polar.cl(0.0), "CD(0)=", polar.cd(0.0))
    print("CL(15deg)=", polar.cl(np.deg2rad(15.0)), "CD(15deg)=", polar.cd(np.deg2rad(15.0)))
    print("CL(20deg)=", polar.cl(np.deg2rad(20.0)), "CD(20deg)=", polar.cd(np.deg2rad(20.0)))
    print("dCN/dalpha @0 (per rad)=", dcn_dalpha_est(polar, 0.0))

    # ---- Choose models (canonical outputs: CL/CD/CT/CN) ----
    oye = OyeDynamicStallCL(c=1.0, Tf0=6.0, use_bergami_cd=True)
    lb = LeishmanBeddoesFirstOrder(c=1.0, alpha_crit_rad=np.deg2rad(15.0))
    iag = IAGDynamicStallCL(c=1.0, ks=0.06, alpha_crit_rad=np.deg2rad(15.0))
    print("IAG ks =", iag.ks)

    models = {
        "NoDS": NoDynamicStall(),
        "Oye": oye,
        "LB-1st": lb,
        "IAG": iag,
    }

    # ---- Pitch hysteresis (no inflow) ----
    dt = 1e-3
    U = 15.0
    n_cycles = 10
    freq_hz = 0.377
    t_end = n_cycles / freq_hz

    for name, m in models.items():
        r = run_pitch_sine(
            model=m,
            polar=polar,
            dt=dt,
            t_end=t_end,
            alpha_mean_deg=18.55,
            alpha_amp_deg=10.45,
            freq_hz=0.377,
            U=U,
        )
        plot_hysteresis(r["alpha_deg"], r["cn"], ylabel="CN", title=f"{name}: CN hysteresis (no inflow)")
        plot_hysteresis(r["alpha_deg"], r["cl"], ylabel="CL", title=f"{name}: CL hysteresis (no inflow)")

    # ---- Pitch hysteresis with optional dynamic inflow wrapper (example for IAG) ----
    if True:
        inflow = DynamicInflowModel(tau_factor=1.0, gain=0.05)
        coupled = CoupledInflowWrapper(ds=iag, inflow=inflow, R=50.0)

        r_c = run_pitch_sine(
            model=coupled,  # wrapper still matches interface because it exposes step-like signature internally
            polar=polar,
            dt=dt,
            t_end=t_end,
            alpha_mean_deg=18.55,
            alpha_amp_deg=10.45,
            freq_hz=0.377,
            U=U,
        )
    plot_hysteresis(r_c["alpha_deg"], r_c["cn"], ylabel="CN", title="IAG: CN hysteresis (with inflow wrapper)")

    # ---- Tier-B energy/work-per-cycle ----
    cfg = TierBConfig(U=15.0, alpha0_deg=18.55, X=0.25, f_hz=0.377, dt=1e-3, c=1.0)

    for name, m in models.items():
        rB = run_tierB(cfg, polar, m)
        print("\n==============================")
        print("CASE:", name)
        print(f"Reduced frequency k = {rB['reduced_frequency_k']:.5f}")
        print(f"Mean work/cycle W = {rB['W_mean']:.6g} J   (std {rB['W_std']:.6g})")
        print("alpha deg range:",
              np.rad2deg(rB["alpha_rad"]).min(),
              np.rad2deg(rB["alpha_rad"]).max())

        print("CL min/max:", rB["cl"].min(), rB["cl"].max())
        print("CD min/max:", rB["cd"].min(), rB["cd"].max())
        print("CT min/max:", rB["ct"].min(), rB["ct"].max())
        print("CT mean:", np.mean(rB["ct"]))

        if name == "IAG":
            a = rB["alpha_rad"]
            cd_static_arr = np.array([polar.cd(ai) for ai in a])
            print("CD_static(min/max over cycle):", cd_static_arr.min(), cd_static_arr.max())
            print("d2cn_raw min/max:", np.nanmin(rB["d2cn_raw"]), np.nanmax(rB["d2cn_raw"]))

        print("Interpretation: W>0 => aero adds energy (negative damping tendency); W<0 => aero removes energy.")
        plot_time_series(
            rB["t"],
            {"ct": rB["ct"], "power": rB["power"]},
            title=f"{name}: CT and Power (Tier-B)"
        )


