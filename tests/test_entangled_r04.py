import unittest
import numpy as np
from mapping_spaces.entangled import qm, frames, r04_physical_qd as r


class R04(unittest.TestCase):
    def test_claim1_L_drops_out(self):
        y = np.array([1.0, 2.5])
        for L in (1e-6, 1.0, 1e3):
            xA, xB = r.lab_tracks_physical(2.0, y, L)
            np.testing.assert_allclose(xA, -y / 2.0, atol=1e-12)
            np.testing.assert_allclose(xB, y / 2.0, atol=1e-12)
        # the physical map agrees with the dimensionless one at Z = L z
        x2, y2, Z2 = r.to_frame_physical(frames.A, frames.LAB, 2.0, 0.3, 1.0, -5.0, 5.0)
        x1, y1, z1 = frames.to_frame(frames.A, frames.LAB, 2.0, 0.3, 1.0, -1.0)
        self.assertAlmostEqual(float(x2), float(x1))
        self.assertAlmostEqual(float(Z2), -5.0)
        self.assertAlmostEqual(r.max_L_in_circle(30e-6), 47.12e-6, delta=0.01e-6)

    def test_claim2_massless_tower(self):
        hbar_c_eVm = r.HBAR * r.C / r.EV
        self.assertAlmostEqual(hbar_c_eVm, 1.9733e-7, delta=1e-10)
        tab = r.tower_table(np.array([1e-12, 1e-10, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]))
        expect = np.array([1.973e5, 1.973e3, 19.73, 1.973, 0.1973, 0.01973, 1.973e-3])
        np.testing.assert_allclose(tab[:, 1], expect, rtol=1e-3)
        th = r.thresholds()
        self.assertAlmostEqual(th["massless_R_at_1eV"], 1.973e-7, delta=1e-10)
        self.assertAlmostEqual(th["massless_R_at_kT"], 7.893e-6, delta=1e-9)
        self.assertAlmostEqual(th["massless_R_at_1keV"], 1.973e-10, delta=1e-13)
        # E_n grows linearly in n
        np.testing.assert_allclose(r.kk_massless_energy(3, 1e-9), 3 * r.kk_massless_energy(1, 1e-9))

    def test_claim3_ring_tower(self):
        self.assertAlmostEqual(r.HBAR**2 / (2 * r.M_E) / r.EV, 3.8100e-20, delta=1e-23)
        tab = r.tower_table(np.array([1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-6, 1e-4]))
        expect = np.array([3.810e4, 381.0, 3.810, 0.03810, 3.810e-4, 3.810e-8, 3.810e-12])
        np.testing.assert_allclose(tab[:, 2], expect, rtol=1e-3)
        th = r.thresholds()
        self.assertAlmostEqual(th["ring_R_at_1eV"], 1.952e-10, delta=1e-13)
        self.assertAlmostEqual(th["ring_R_at_kT"], 1.235e-9, delta=1e-12)
        self.assertAlmostEqual(th["ring_R_at_1keV"], 6.173e-12, delta=1e-15)
        # massive KK vs ring: < 4 % at 1e-12 m, < 1e-6 for R >= 1e-9 m
        rel = abs(tab[:, 3] - tab[:, 2]) / tab[:, 2]
        self.assertLess(rel[0], 0.04)
        self.assertTrue(np.all(rel[3:] < 1e-6))
        self.assertAlmostEqual(r.crossover_radius(), 1.931e-13, delta=1e-16)
        Rc = r.crossover_radius()
        self.assertAlmostEqual(float(r.kk_massless_energy(1, Rc) / r.ring_energy(1, Rc)), 1.0)
        # ring E_n ~ n^2
        np.testing.assert_allclose(r.ring_energy(2, 1e-9), 4 * r.ring_energy(1, 1e-9))

    def test_claim4_spreading_excludes_free_motion(self):
        self.assertAlmostEqual(float(r.spreading_time(30e-6)), 7.77e-6, delta=0.01e-6)
        self.assertAlmostEqual(float(r.spreading_time(1.97e-10)), 3.35e-16, delta=0.01e-16)
        ratio = float(r.placement_persistence_ratio(30e-6))
        self.assertAlmostEqual(ratio, 2.57e5, delta=0.01e5)
        self.assertGreater(ratio, 1e5)
        self.assertLess(float(r.spreading_time(30e-6)), r.L_B6A_STORED_S[0])
        self.assertLess(float(r.spreading_time(30e-6)), r.L_B6B_STORED_S)
        # narrower packets spread faster (the R^2/hbar figure is the most generous)
        self.assertLess(float(r.spreading_time(30e-6, delta=3e-6)), float(r.spreading_time(30e-6)))

    def test_claim5_yukawa_deviations(self):
        tab = r.yukawa_sensitivity(lam=30e-6, alpha=1.0, r=(52e-6, 1e-3, 3e-3))
        np.testing.assert_allclose(tab[:, 1], [0.1767, 3.34e-15, 3.72e-44], rtol=5e-3)
        np.testing.assert_allclose(tab[:, 2], [0.4830, 1.15e-13, 3.76e-42], rtol=5e-3)
        # closed forms agree with explicit subtraction where nothing underflows
        dv, df = r.deviations_by_subtraction(52e-6, 1.0, 1.0, 1.0, 30e-6)
        self.assertAlmostEqual(float(dv), 0.17669, places=5)
        self.assertAlmostEqual(float(df), 0.48296, places=5)
        # ledger numbers as recorded
        self.assertEqual(r.L_C1A_LAMBDA_MAX, 38.6e-6)
        self.assertEqual(r.L_C1A_R_STAR_MAX, 30e-6)
        self.assertEqual(r.L_C1B_LAMBDA_EXCLUDED, 48e-6)
        self.assertGreater(r.L_C1B_LAMBDA_EXCLUDED, r.L_C1A_LAMBDA_MAX)
        # Yukawa reduces to Newton when alpha = 0
        np.testing.assert_allclose(r.yukawa_potential(1e-4, 2.0, 3.0, 0.0, 30e-6),
                                   r.newton_potential(1e-4, 2.0, 3.0))

    def test_claim6_allowed_window(self):
        w = r.allowed_window()
        self.assertEqual(w["R_max_gravity"], 30e-6)
        self.assertAlmostEqual(w["L_max_gravity"], 47.12e-6, delta=0.01e-6)
        self.assertAlmostEqual(w["R_max_free_massless_1keV"], 1.973e-10, delta=1e-13)
        self.assertAlmostEqual(w["R_max_free_ring_1keV"], 6.173e-12, delta=1e-15)
        self.assertTrue(bool(w["free_motion_excluded"]))
        self.assertAlmostEqual(w["ring_E1_eV_at_R_gravity"], 4.23e-11, delta=0.01e-11)
        self.assertAlmostEqual(w["force_deviation_1mm_at_R_gravity"], 1.15e-13, delta=0.01e-13)
        self.assertLess(w["force_deviation_1mm_at_R_gravity"], 1e-12)

    def test_claim7_spin_as_M1(self):
        s = r.singlet_statistics()
        self.assertAlmostEqual(s["S"], qm.TSIRELSON, places=12)
        self.assertAlmostEqual(s["S"], 2.8284, places=4)
        np.testing.assert_allclose(s["marginals"], 0.5, atol=1e-12)
        self.assertTrue(s["ledger_S_within_bound"])
        self.assertLess(r.L_A6D_S, qm.TSIRELSON)
        self.assertLess(r.L_A6E_S, qm.TSIRELSON)


if __name__ == "__main__":
    unittest.main()
