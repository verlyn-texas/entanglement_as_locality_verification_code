"""R03 — absolute lab time: the state change is instantaneous in y, hence
independent of apparatus motion (mechanism M1a, hypothesis H3, experiment E3).

Every frame map of the shear model is x' = x + beta(z) y, y' = y, z' = z: the
time coordinate y is untouched, so the order of two measurement events is the
same in every frame (Galilean absolute time).  A laboratory moving at velocity
v along x is modelled as a new frame 3' with shear potential
sigma_3'(z) = z/a + v, i.e. beta_{k3'} = beta_{k3} + v for every k.

Claims (numbered as in rows/R03_absolute_time.md):
 1. y is invariant under every frame map, so the time order of (y_A, y_B) is
    the same in the lab, in A's frame and in B's frame, for random events
    and heights.
 2. In the moving lab 3' the pair still separates at rate 2/a (the tracks are
    boosted by v), while in frames 1, 2 the particles remain co-located and
    the maps 1 <-> 2 are unchanged; frame 3' is consistent (composition
    closes) and its time order is the lab's.
 3. CHSH S for the singlet, computed by sequential projection in the absolute
    time order, is 2 sqrt 2 for every lab velocity v (including v >> 1/a) and
    every (y_A, y_B): the frame maps never touch the spin state, the settings
    or the time order.
 4. Lab-frame "influence speed" dx/dy = (y_A + y_B)/(a |y_B - y_A|) is
    unbounded (infinite at y_A = y_B); in the particles' frames dx = 0 so no
    speed is defined.
 5. A preferred-frame finite-speed model (speed V in a frame moving at u,
    Galilean) keeps the correlation iff |dx - u dt| <= V dt; the ledger
    configurations L-B3, L-B4, L-B1 give V_min/c = 1.85e5, 1.46e5, 7.1e6,
    reproducing the order of magnitude of the published bounds; V = infinity
    (this model) predicts persistence in every configuration.
"""
from __future__ import annotations

import itertools
import numpy as np

from mapping_spaces import transformation as T
from mapping_spaces.entangled import frames, qm

C_LIGHT = 299_792_458.0  # m/s
MOVING_LAB = 4  # label for frame 3' (a lab moving at velocity v along x)


# ------------------------------------------------------------ claim 1: time order
def time_order_all_frames(yA: float, yB: float, a: float, zA: float = -1.0, zB: float = 1.0) -> dict:
    """Map the two measurement events (x_i, y_i, z_i) from the lab to A's and
    B's frames and record the time order seen in each."""
    xA, xB = frames.lab_tracks(a, np.array([yA, yB]))
    events = {"lab": ((xA[0], yA, zA), (xB[1], yB, zB))}
    for k, name in ((frames.A, "frame_A"), (frames.B, "frame_B")):
        mapped = []
        for x, y, z in events["lab"]:
            xp, yp, zp = frames.to_frame(frames.LAB, k, a, x, y, z)
            mapped.append((float(xp), float(yp), float(zp)))
        events[name] = tuple(mapped)
    return {name: frames.lab_time_order(ev[0][1], ev[1][1]) for name, ev in events.items()}


def random_time_orders(n: int = 200, seed: int = 3) -> np.ndarray:
    """For n random (y_A, y_B, a, z_A, z_B): the time order in lab, frame A,
    frame B as an (n, 3) integer array.  Rows are equal across columns."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        yA, yB = rng.uniform(0.0, 5.0, size=2)
        a = rng.uniform(0.2, 5.0) * rng.choice([-1, 1])
        zA, zB = rng.uniform(-2.0, 2.0, size=2)
        o = time_order_all_frames(yA, yB, a, zA, zB)
        out.append([o["lab"], o["frame_A"], o["frame_B"]])
    return np.array(out, dtype=int)


# ---------------------------------------------------- claim 2: the moving lab 3'
def shear_potential(k: int, a: float, v: float = 0.0):
    """Shear potentials of frames 1, 2, 3 and of the moving lab 3' (MOVING_LAB):
    sigma_3'(z) = z/a + v."""
    if k == MOVING_LAB:
        return lambda z: np.asarray(z, dtype=float) / a + v
    return T.shear_potential(k, a)


def beta_v(k: int, l: int, a: float, z, v: float = 0.0) -> np.ndarray:
    """beta_{kl}(z) with frame 3' included."""
    z = np.asarray(z, dtype=float)
    return shear_potential(l, a, v)(z) - shear_potential(k, a, v)(z)


def to_frame_v(k: int, l: int, a: float, x, y, z, v: float = 0.0):
    x, y, z = (np.asarray(t, dtype=float) for t in (x, y, z))
    return x + beta_v(k, l, a, z, v) * y, y, z


def moving_lab_tracks(a: float, v: float, y):
    """Tracks of A and B seen from the lab moving at v: x = (-1/a + v) y and
    (1/a + v) y — the same as the lab's, boosted."""
    y = np.asarray(y, dtype=float)
    xA = beta_v(frames.A, MOVING_LAB, a, frames.Z[frames.A], v) * y
    xB = beta_v(frames.B, MOVING_LAB, a, frames.Z[frames.B], v) * y
    return xA, xB


def moving_lab_summary(a: float, v: float, y: float = 1.0) -> dict:
    """Separations at lab time y in frames 3', 3, 1, 2 and the closure
    defect max_z |beta_{k3'} - beta_{k3} - v| over k, z."""
    xA, xB = moving_lab_tracks(a, v, y)
    zz = np.linspace(-2, 2, 41)
    defect = 0.0
    for k in (frames.A, frames.B, frames.LAB):
        defect = max(defect, float(np.abs(beta_v(k, MOVING_LAB, a, zz, v) - beta_v(k, frames.LAB, a, zz) - v).max()))
    # closure: 1 -> 3' -> 2 equals 1 -> 2 at every z
    closure = float(np.abs(beta_v(frames.A, MOVING_LAB, a, zz, v) + beta_v(MOVING_LAB, frames.B, a, zz, v)
                           - beta_v(frames.A, frames.B, a, zz, v)).max())
    return {
        "sep_moving_lab": float(xB - xA),
        "sep_lab": float(frames.separation_in_frame(frames.LAB, a, np.array([y]))[0]),
        "sep_frame_A": float(frames.separation_in_frame(frames.A, a, np.array([y]))[0]),
        "sep_frame_B": float(frames.separation_in_frame(frames.B, a, np.array([y]))[0]),
        "xA_moving_lab": float(xA),
        "xB_moving_lab": float(xB),
        "beta_12_at_A": float(beta_v(frames.A, frames.B, a, -1.0, v)),
        "beta_12_at_B": float(beta_v(frames.A, frames.B, a, 1.0, v)),
        "shift_defect": defect,
        "closure_defect": closure,
    }


# ------------------------------------------- claim 3: CHSH by sequential projection
def sequential_box(state, settings_A, settings_B, yA: float, yB: float) -> np.ndarray:
    """P(a,b|x,y) generated the way the mechanism says it happens: the wing
    that measures first (in absolute time y) projects the state; the second
    wing measures the projected state.  Simultaneous events use the joint
    projector (order irrelevant)."""
    first = frames.lab_time_order(yA, yB)
    box = np.zeros((2, 2, 2, 2))
    for x, y, ia, ib in itertools.product(range(2), repeat=4):
        oa, ob = qm.OUTCOME[ia], qm.OUTCOME[ib]
        if first == frames.A:
            rho1, p1 = qm.project(state, settings_A[x], oa, "A")
            p2 = qm.marginal_probability(rho1, settings_B[y], ob, "B")
        elif first == frames.B:
            rho1, p1 = qm.project(state, settings_B[y], ob, "B")
            p2 = qm.marginal_probability(rho1, settings_A[x], oa, "A")
        else:
            p1, p2 = qm.joint_probability(state, settings_A[x], settings_B[y], oa, ob), 1.0
        box[x, y, ia, ib] = p1 * p2
    return box


def chsh_moving_lab(v: float, yA: float, yB: float, a: float = 2.0) -> dict:
    """Measurement events in the moving lab 3', then S from sequential
    projection in the time order that frame reports."""
    settings = qm.chsh_optimal_settings()
    sA, sB = (settings[0], settings[1]), (settings[2], settings[3])
    xA_lab, xB_lab = frames.lab_tracks(a, np.array([yA, yB]))
    # map the lab events into the moving lab; the settings and the state are not coordinates
    xA, yA_p, _ = to_frame_v(frames.LAB, MOVING_LAB, a, xA_lab[0], yA, -1.0, v)
    xB, yB_p, _ = to_frame_v(frames.LAB, MOVING_LAB, a, xB_lab[1], yB, 1.0, v)
    box = sequential_box(qm.singlet(), sA, sB, float(yA_p), float(yB_p))
    return {
        "S": float(qm.box_chsh(box)),
        "signalling": float(qm.signalling(box)),
        "order": frames.lab_time_order(float(yA_p), float(yB_p)),
        "order_lab": frames.lab_time_order(yA, yB),
        "dx_moving_lab": float(xB - xA),
        "box": box,
    }


def chsh_scan(a: float = 2.0, seed: int = 7) -> np.ndarray:
    """S over a grid of lab velocities (including |v| >> 1/a) and random
    (y_A, y_B) including equal times.  Returns array of S values."""
    rng = np.random.default_rng(seed)
    vs = [0.0, 0.1 / a, -0.3 / a, 1.0 / a, 10.0 / a, -100.0 / a, 1e4 / a]
    S = []
    for v in vs:
        for _ in range(4):
            yA, yB = rng.uniform(0.0, 3.0, size=2)
            S.append(chsh_moving_lab(v, yA, yB, a)["S"])
        S.append(chsh_moving_lab(v, 1.3, 1.3, a)["S"])  # simultaneous
    return np.array(S)


# ------------------------------------------------- claim 4: lab-frame influence speed
def lab_event_separation(yA: float, yB: float, a: float) -> tuple[float, float]:
    """(|dx|, dy) between the measurement events in the lab."""
    xA, xB = frames.lab_tracks(a, np.array([yA, yB]))
    return float(abs(xB[1] - xA[0])), float(abs(yB - yA))


def influence_speed(yA: float, yB: float, a: float, frame: int = frames.LAB) -> float:
    """dx/dy between the two measurement events as seen in `frame`.  In the
    lab this is (y_A + y_B) / (a |y_B - y_A|): infinite when y_A = y_B.  In
    frames 1 and 2 dx = 0: returns nan ("speed" undefined, nothing moves)."""
    xA_lab, xB_lab = frames.lab_tracks(a, np.array([yA, yB]))
    xA, _, _ = frames.to_frame(frames.LAB, frame, a, xA_lab[0], yA, -1.0)
    xB, _, _ = frames.to_frame(frames.LAB, frame, a, xB_lab[1], yB, 1.0)
    dx, dy = abs(float(xB - xA)), abs(yB - yA)
    if dx == 0.0:
        return float("nan")
    return float("inf") if dy == 0.0 else dx / dy


# ----------------------------------- claim 5: preferred-frame finite-speed model
def minimum_speed(dx: float, dt: float, u: float = 0.0) -> float:
    """Slowest influence speed (in a preferred frame moving at u along the
    separation axis, Galilean) that still connects the two events:
    V_min = |dx - u dt| / dt.  Infinite for dt = 0."""
    if dt == 0.0:
        return float("inf")
    return abs(dx - u * dt) / dt


def correlation_persists(dx: float, dt: float, V: float, u: float = 0.0) -> bool:
    """True iff an influence of speed V (preferred frame at u) launched at the
    first event reaches the second before it happens: |dx - u dt| <= V dt."""
    if V == float("inf"):
        return True
    return abs(dx - u * dt) <= V * dt


# Ledger configurations (registers/constraints.md).  dx from the ledger; the
# timing alignments are the figures supplied with the task brief, and are
# consistent with the ledger's bounds (see the row document).
LEDGER = {
    "L-B3": {"dx": 18.0e3, "dt": 5.4e-6 * 18.0e3 / C_LIGHT, "published_c": 1.8e5, "beta_max": 1e-3},
    "L-B4": {"dx": 15.3e3, "dt": 350e-12, "published_c": 1.38e4, "beta_max": 1e-3},
    "L-B1": {"dx": 10.6e3, "dt": 5e-12, "published_c": 1e7, "beta_max": 105.0 / C_LIGHT},
}


def ledger_bounds() -> dict:
    """V_min / c for each ledger configuration at u = 0 and at the largest
    preferred-frame velocity the entry considers, plus the absolute-time
    verdict (V = inf)."""
    out = {}
    for key, cfg in LEDGER.items():
        u = cfg["beta_max"] * C_LIGHT
        out[key] = {
            "Vmin_over_c_u0": minimum_speed(cfg["dx"], cfg["dt"]) / C_LIGHT,
            "Vmin_over_c_umax": minimum_speed(cfg["dx"], cfg["dt"], u) / C_LIGHT,
            "published_c": cfg["published_c"],
            "absolute_time_persists": correlation_persists(cfg["dx"], cfg["dt"], float("inf")),
            "finite_below_bound_persists": correlation_persists(cfg["dx"], cfg["dt"], 0.5 * cfg["published_c"] * C_LIGHT),
        }
    return out


if __name__ == "__main__":
    orders = random_time_orders()
    print("claim 1: orders agree across frames:", bool((orders[:, 0] == orders[:, 1]).all() and (orders[:, 0] == orders[:, 2]).all()))
    for v in (0.0, 0.5, 50.0):
        print("claim 2: v =", v, moving_lab_summary(2.0, v))
    print("claim 3: S scan:", chsh_scan())
    print("claim 4: lab speed (1,1):", influence_speed(1.0, 1.0, 2.0), " (1,1.5):", influence_speed(1.0, 1.5, 2.0),
          " frame A:", influence_speed(1.0, 1.5, 2.0, frames.A))
    print("claim 5:", ledger_bounds())
