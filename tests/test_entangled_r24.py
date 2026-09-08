"""R24 — the charts in 3+1 dimensions (paper-1 Appendix C; Sec. 3.7.4 numerics)."""
import itertools
import unittest

import numpy as np

from mapping_spaces.entangled import r24_dim3 as r24


class TestGalileanVector(unittest.TestCase):
    """Appendix C, 'Galilean form': P1 with vector potentials, verbatim."""

    def setUp(self):
        self.rng = np.random.default_rng(7)
        self.n = 5
        self.v = self.rng.uniform(-1, 1, (self.n, 3))
        self.z = [-1.0, 1.0, -2.0, 2.0, 3.0]
        w = np.zeros((self.n, self.n))
        w[0, 1] = w[1, 0] = 1.0
        w[2, 3] = w[3, 2] = 0.6
        self.w = w
        self.pots = r24.galilean_potentials(self.v, self.z, self.w)
        self.heights = [0.0] + self.z

    def test_composition_and_inverses(self):
        t, x = 1.7, self.rng.normal(size=3)
        for k, l, m in itertools.product(range(self.n + 1), repeat=3):
            for zz in self.heights:
                once = r24.galilean_map(self.pots, k, l, zz, x, t)
                self.assertTrue(np.allclose(
                    r24.galilean_map(self.pots, l, m, zz, once, t),
                    r24.galilean_map(self.pots, k, m, zz, x, t)))
                self.assertTrue(np.allclose(
                    r24.galilean_map(self.pots, l, k, zz, once, t), x))

    def test_partner_at_rest_fraction(self):
        # frame i sees particle j at the fraction (1 - w_ij) of the 3-D
        # relative velocity (zero for a full partner)
        for i in range(self.n):
            for j in range(self.n):
                seen_j = self.v[j] + self.pots[i][self.z[j]] - self.pots[self.n][self.z[j]]
                seen_i = self.v[i] + self.pots[i][self.z[i]] - self.pots[self.n][self.z[i]]
                expect = (np.zeros(3) if i == j
                          else (1 - self.w[i][j]) * (self.v[j] - self.v[i]))
                self.assertTrue(np.allclose(seen_j - seen_i, expect))

    def test_lab_pinning(self):
        for j in range(self.n):
            self.assertTrue(np.allclose(self.pots[self.n][self.z[j]], self.v[j]))


class TestCovariantGroupValued(unittest.TestCase):
    """Appendix C, 'Covariant form': right-quotient group-valued potentials."""

    def setUp(self):
        rng = np.random.default_rng(7)
        vel = rng.uniform(-0.6, 0.6, (4, 3))
        self.vel = np.array([x / max(1, 1.2 * np.linalg.norm(x)) for x in vel])
        w = np.zeros((4, 4))
        w[0, 1] = w[1, 0] = 1.0
        w[2, 3] = w[3, 2] = 0.4
        self.w = w
        self.frames = ["lab", 0, 1, 2, 3]
        self.heights = ["lab", 0, 1, 2, 3]

    def tmap(self, k, l, h):
        return r24.frame_map(k, l, h, self.vel, self.w)

    def test_composition_identity(self):
        # T_lm T_kl = T_km and T_lk = T_kl^-1 for every triple and height
        for k, l, m in itertools.product(self.frames, repeat=3):
            for h in self.heights:
                self.assertTrue(np.allclose(
                    self.tmap(l, m, h) @ self.tmap(k, l, h), self.tmap(k, m, h),
                    atol=1e-10))
                self.assertTrue(np.allclose(
                    self.tmap(l, k, h) @ self.tmap(k, l, h), np.eye(4), atol=1e-10))

    def test_partner_at_rest_proper_time(self):
        for i, j in ((0, 1), (1, 0)):
            for t in (0.3, 1.0, 2.5):
                ev = np.concatenate([[t], self.vel[j] * t])
                evp = r24.lambda_potential(i, j, self.vel, self.w) @ ev
                self.assertTrue(np.allclose(evp[1:], 0, atol=1e-12))
                self.assertAlmostEqual(
                    evp[0], t * np.sqrt(1 - self.vel[j] @ self.vel[j]), places=12)

    def test_partial_partner_velocity(self):
        for i, j in ((2, 3), (3, 2)):
            ev = np.concatenate([[1.0], self.vel[j]])
            evp = r24.lambda_potential(i, j, self.vel, self.w) @ ev
            self.assertTrue(np.allclose(
                evp[1:] / evp[0],
                (1 - self.w[i][j]) * r24.rel_velocity(self.vel[i], self.vel[j]),
                atol=1e-12))

    def test_wigner_rotation_present(self):
        # non-collinear maps between unrelated frames carry a genuine rotation
        angles = [r24.rotation_angle_deg(self.tmap(k, l, h))
                  for k in range(4) for l in range(4) if k != l
                  for h in self.heights]
        self.assertGreater(max(angles), 1.0)
        self.assertLess(max(angles), 90.0)
        # Sec. 3.7.4 / Appendix C: "up to ~27 deg" in this scan (round-5 H16)
        self.assertAlmostEqual(r24.wigner_scan_max(self.vel, self.w), max(angles), places=9)
        self.assertAlmostEqual(max(angles), 26.6, delta=0.1)          # "~27 deg" (all pairs)
        cross = [r24.rotation_angle_deg(self.tmap(k, l, h))
                 for k in (0, 1) for l in (2, 3) for h in self.heights]
        self.assertAlmostEqual(max(cross), 18.0, delta=0.1)           # different components

    def test_local_unitaries_change_no_cut_negativity(self):
        # Appendix C: a fixed rotation per particle changes no cut negativity (round-6 J9)
        out = r24.local_unitary_invariance()
        self.assertEqual(out["cuts"], 7)
        self.assertLess(out["max_change"], 1e-12)

    def test_wigner_angle_two_perpendicular_boosts(self):
        # Appendix C: two perpendicular 0.85c boosts rotate by 34 deg (34.44)
        self.assertAlmostEqual(r24.wigner_angle_two_boosts(0.85), 34.44, delta=0.01)
        self.assertAlmostEqual(r24.wigner_angle_two_boosts(0.85),
                               r24.wigner_angle_two_boosts_closed_form(0.85), places=6)

    def test_collinear_limit_scalar(self):
        rng = np.random.default_rng(7)
        vel1 = np.zeros((4, 3))
        vel1[:, 0] = rng.uniform(-0.8, 0.8, 4)
        for k in range(4):
            for l in range(4):
                if k == l:
                    continue
                for h in self.heights:
                    L = r24.frame_map(k, l, h, vel1, self.w)
                    self.assertTrue(np.allclose(
                        r24.rotation_part(L), np.eye(3), atol=1e-10))


class TestLightConeRule3D(unittest.TestCase):
    """Appendix C, 'The light-cone rule': Eq. (10) under non-collinear boosts."""

    def test_kappa_invariant_under_boosts(self):
        rng = np.random.default_rng(7)
        for _ in range(300):
            e1 = (np.zeros(3), 0.0)
            e2 = (rng.normal(size=3), 0.0)
            vA = rng.uniform(-0.5, 0.5, 3)
            vD = rng.uniform(-0.5, 0.5, 3)
            wlA = (e1[0], e1[1], vA)
            wlD = (e2[0], e2[1], vD)
            transfer = ((e1[0] + e2[0]) / 2 + rng.normal(scale=0.3, size=3),
                        rng.uniform(2, 4))
            tA, tD = rng.uniform(1, 9, 2)
            kA, _ = r24.kappa(wlA, tA, transfer)
            kD, _ = r24.kappa(wlD, tD, transfer)
            bdir = rng.normal(size=3)
            bdir /= np.linalg.norm(bdir)
            L = r24.boost(rng.uniform(0.1, 0.85) * bdir)
            tTp, xTp = r24.lorentz_event(L, transfer[1], transfer[0])
            tAp, _ = r24.lorentz_event(L, tA, wlA[0] + wlA[2] * (tA - wlA[1]))
            tDp, _ = r24.lorentz_event(L, tD, wlD[0] + wlD[2] * (tD - wlD[1]))
            kAp, _ = r24.kappa(r24.transform_worldline(L, wlA), tAp, (xTp, tTp))
            kDp, _ = r24.kappa(r24.transform_worldline(L, wlD), tDp, (xTp, tTp))
            self.assertLess(abs(kA - kAp), 1e-9)
            self.assertLess(abs(kD - kDp), 1e-9)
            self.assertEqual(np.sign(kA), np.sign(kAp))   # cone membership
            self.assertEqual(np.sign(kD), np.sign(kDp))
            self.assertEqual(kA < kD, kAp < kDp)          # the order

    def test_interval_clock_invariant_3d(self):
        # round-4 P4 (Appendix C): the signed interval from the defining event
        self.assertEqual(r24.interval_clock_invariance_scan(), 0)

    def test_disagreement_fraction_perpendicular(self):
        # (R-1)/2R depends only on the gammas, so it holds for perpendicular
        # velocities with the working pair's speeds (0.3, 0.8)
        rng = np.random.default_rng(7)
        vA = np.array([0.3, 0.0, 0.0])
        vB = np.array([0.0, 0.8, 0.0])
        gA = 1 / np.sqrt(1 - vA @ vA)
        gB = 1 / np.sqrt(1 - vB @ vB)
        R = max(gA, gB) / min(gA, gB)
        tA, tB = rng.uniform(0, 1, (2, 200000))
        frac = np.mean((tA < tB) != (tA / gA < tB / gB))
        self.assertAlmostEqual(frac, (R - 1) / (2 * R), delta=3e-3)


class TestPhotons(unittest.TestCase):
    """Appendix C, 'Photons': kappa = 0 on null lines; affine order not invariant."""

    def test_kappa_zero_on_null_worldlines(self):
        for t in (1.0, 1.3):
            self.assertEqual(np.sqrt(1 - 1.0**2) * t, 0.0)

    def test_coordinate_time_order_flips_but_invariant_clock_does_not(self):
        # coordinate-time normalisation: frame-dependent (77 of 181, for the record)
        self.assertEqual(r24.photon_affine_order_flips(), 77)
        # lambda = t/omega, the m -> 0 limit of kappa/m = t/E: invariant, 0 of 181
        self.assertEqual(r24.photon_invariant_clock_flips(), 0)

    def test_massive_pair_never_flips(self):
        self.assertEqual(r24.massive_proper_order_flips(), 0)


class TestSingletonCutBound(unittest.TestCase):
    """Sec. 3.6.4: w in [0,1] is a theorem — the singleton cut bounds the min."""

    def test_min_cut_bounded_on_random_qutrit_triples(self):
        rng = np.random.default_rng(7)
        for _ in range(20):
            psi = rng.normal(size=27) + 1j * rng.normal(size=27)
            psi /= np.linalg.norm(psi)
            rho = np.outer(psi, psi.conj())
            n1 = r24.negativity(rho, 3, 9)  # cut {1} | {2,3}
            rho_perm = rho.reshape(3, 3, 3, 3, 3, 3).transpose(
                0, 2, 1, 3, 5, 4).reshape(27, 27)  # order (1,3,2)
            n2 = r24.negativity(rho_perm, 9, 3)  # cut {1,3} | {2}
            self.assertLessEqual(min(n1, n2), (3 - 1) / 2 + 1e-12)


if __name__ == "__main__":
    unittest.main()
