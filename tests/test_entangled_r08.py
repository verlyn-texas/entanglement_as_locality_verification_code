import unittest
import numpy as np
from mapping_spaces.entangled import qm, frames, r08_epr_positions as r


HB, ME = r.HBAR, r.M_E


class R08(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = r.summary()

    # ---------------------------------------------------------- claim 1
    def test_claim1_tmsv_variances_and_physicality(self):
        for rr in (0.0, 0.5, 1.0, 2.0):
            for sigma0, hbar in ((1 / np.sqrt(2), 1.0), (10e-9, HB)):
                V = r.tmsv_cov(rr, sigma0, hbar)
                sp = r.sigma_p(sigma0, hbar)
                self.assertAlmostEqual(r.var_of(V, r.X_MINUS) / (2 * sigma0**2), np.exp(-2 * rr), places=12)
                self.assertAlmostEqual(r.var_of(V, r.X_PLUS) / (2 * sigma0**2), np.exp(2 * rr), places=12)
                self.assertAlmostEqual(r.var_of(V, r.P_PLUS) / (2 * sp**2), np.exp(-2 * rr), places=12)
                self.assertAlmostEqual(r.var_of(V, r.P_MINUS) / (2 * sp**2), np.exp(2 * rr), places=12)
                self.assertAlmostEqual(V[0, 0] / sigma0**2, np.cosh(2 * rr), places=12)
                self.assertAlmostEqual(V[2, 2] / sigma0**2, np.cosh(2 * rr), places=12)
                # bona fide state: both symplectic eigenvalues equal hbar/2 (pure)
                np.testing.assert_allclose(r.symplectic_eigenvalues(V) / (hbar / 2), 1.0, rtol=1e-10)
                np.testing.assert_allclose(V, V.T)
        # dimensionless limit reproduces the CV-literature vacuum variance 1/2
        np.testing.assert_allclose(r.tmsv_cov(0.0), 0.5 * np.eye(4), atol=1e-14)

    # ---------------------------------------------------------- claim 2
    def test_claim2_reid_criterion(self):
        for rr in (0.5, 1.0, 2.0):
            V = r.tmsv_cov(rr)
            out = r.reid(V)
            self.assertAlmostEqual(out["var_inf_x"], 0.5 / np.cosh(2 * rr), places=12)
            self.assertAlmostEqual(out["var_inf_p"], 0.5 / np.cosh(2 * rr), places=12)
            self.assertAlmostEqual(out["ratio"], 1 / np.cosh(2 * rr) ** 2, places=12)
            self.assertTrue(out["satisfied"])
            # SI version: same ratio, product in J^2 s^2
            Vsi = r.tmsv_cov(rr, 10e-9, HB)
            self.assertAlmostEqual(r.reid(Vsi, HB)["ratio"], out["ratio"], places=10)
        self.assertAlmostEqual(self.s["reid_ratio_r0.5"], 0.4200, places=4)
        self.assertAlmostEqual(self.s["reid_ratio_r1.0"], 0.07065, places=5)
        self.assertAlmostEqual(self.s["reid_ratio_r2.0"], 0.001341, places=6)
        # r = 0: equality, not satisfied
        self.assertAlmostEqual(self.s["reid_ratio_r0.0"], 1.0, places=12)
        self.assertFalse(r.reid(r.tmsv_cov(0.0))["satisfied"])

    # ---------------------------------------------------------- claim 3
    def test_claim3_duan_simon_and_ppt(self):
        for rr, dsum in ((0.5, 0.7358), (1.0, 0.2707), (2.0, 0.03663)):
            V = r.tmsv_cov(rr)
            out = r.duan(V, 1 / np.sqrt(2))
            self.assertAlmostEqual(out["sum"], 2 * np.exp(-2 * rr), places=12)
            self.assertAlmostEqual(out["sum"], dsum, places=4)
            self.assertTrue(out["satisfied"])
            self.assertAlmostEqual(r.ppt_min_symplectic(V), 0.5 * np.exp(-2 * rr), places=10)
            Vsi = r.tmsv_cov(rr, 10e-9, HB)
            self.assertAlmostEqual(r.duan(Vsi, 10e-9, HB)["sum"], out["sum"], places=10)
            self.assertAlmostEqual(r.ppt_min_symplectic(Vsi) / (HB / 2), np.exp(-2 * rr), places=10)
        self.assertAlmostEqual(self.s["duan_sum_r0.0"], 2.0, places=12)
        self.assertAlmostEqual(self.s["nu_minus_r0.0"], 0.5, places=12)

    # ---------------------------------------------------------- claim 4
    def test_claim4_frame_dictionary(self):
        a, m = 2.0, ME
        for y in (0.0, 0.7, 3.0):
            mean_lab = r.lab_mean(a, m, y)
            np.testing.assert_allclose(mean_lab[[0, 2]], frames.lab_tracks(a, y), atol=1e-30)
            for k in (frames.A, frames.B):
                mk = r.mean_in_frame(mean_lab, k, a, m, y)
                np.testing.assert_allclose(mk, 0.0, atol=1e-25)   # both at x = 0, p = 0
            # frame 3 -> 3 is the identity
            np.testing.assert_allclose(r.mean_in_frame(mean_lab, frames.LAB, a, m, y), mean_lab)
        V = r.tmsv_cov(1.0, 10e-9, HB)
        for k in (frames.A, frames.B, frames.LAB):
            np.testing.assert_array_equal(r.cov_in_frame(V, k), V)
        # co-location rms in the particles' frame
        self.assertAlmostEqual(r.colocation_rms(1.0, 10e-9) * 1e9, 5.203, places=3)
        self.assertAlmostEqual(r.colocation_rms(1.0, 10e-9) ** 2, r.var_of(V, r.X_MINUS), places=25)
        # Gaussian mean evolution reproduces the lab tracks x = -/+ y/a
        for t in (0.3, 1.5):
            _, mean_t = r.evolve(V, r.lab_mean(a, m, 0.0), t, m)
            np.testing.assert_allclose(mean_t, r.lab_mean(a, m, t), rtol=1e-12)

    # ---------------------------------------------------------- claim 5
    def test_claim5_free_evolution_invariants(self):
        rr, sigma0, m = 1.0, 10e-9, ME
        t2 = r.doubling_time(rr, sigma0, m)
        times = np.array([0, 0.25, 0.5, 1, 2, 4, 20]) * t2
        tab = r.free_flight_table(rr, sigma0, m, times)
        np.testing.assert_allclose(tab["var_x_rel"], tab["var_x_rel_closed"], rtol=1e-10)
        for key in ("var_p_plus", "var_p_minus", "var_back_evolved", "nu_minus", "sympl_min"):
            np.testing.assert_allclose(tab[key], tab[key][0], rtol=1e-10, err_msg=key)
        self.assertAlmostEqual(tab["nu_minus"][0] / (HB / 2), np.exp(-2 * rr), places=10)
        self.assertAlmostEqual(tab["sympl_min"][0] / (HB / 2), 1.0, places=10)
        self.assertAlmostEqual(tab["cov_xrel_prel"][0], 0.0)
        # single-particle uncertainty product never below hbar^2/4, and grows
        self.assertTrue(np.all(tab["uncert_1"] >= HB**2 / 4 * (1 - 1e-12)))
        self.assertTrue(np.all(np.diff(tab["uncert_1"]) > 0))
        # the symplectic map is symplectic
        S = r.free_symplectic(t2, m)
        np.testing.assert_allclose(S @ r.OMEGA @ S.T, r.OMEGA, atol=1e-12)
        # fixed-quadrature witnesses are lost while nu_- is not
        self.assertGreater(tab["duan_sum"][-1], 2.0)
        self.assertGreater(tab["reid_ratio"][-1], 1.0)

    # ---------------------------------------------------------- claim 6
    def test_claim6_electron_numbers(self):
        s = self.s
        rr, sigma0, m = 1.0, 10e-9, ME
        self.assertAlmostEqual(s["t2"] / 4.05e-13, 1.0, places=2)
        self.assertAlmostEqual(s["t2_r0"] / s["t2"], np.exp(2 * rr), places=10)
        self.assertAlmostEqual(s["t2_r0"] / 2.99e-12, 1.0, places=2)
        self.assertAlmostEqual(s["t_duan"] / 8.36e-13, 1.0, places=2)
        self.assertAlmostEqual(s["t_reid"] / 9.81e-13, 1.0, places=2)
        self.assertAlmostEqual(s["sigma0_1ns"] / 0.497e-6, 1.0, places=2)
        self.assertAlmostEqual(s["t2_he4_1um"] / 2.95e-5, 1.0, places=2)
        # definitions: doubling, Duan sum = 2, Reid ratio = 1 at the quoted times
        rms0 = np.sqrt(r.x_rel_variance_closed_form(0.0, rr, sigma0, m))
        rms2 = np.sqrt(r.x_rel_variance_closed_form(s["t2"], rr, sigma0, m))
        self.assertAlmostEqual(rms2 / rms0, 2.0, places=10)
        self.assertAlmostEqual(rms0 * 1e9, 5.203, places=3)
        V0 = r.tmsv_cov(rr, sigma0, HB)
        Vd, _ = r.evolve(V0, np.zeros(4), s["t_duan"], m)
        self.assertAlmostEqual(r.duan(Vd, sigma0, HB)["sum"], 2.0, places=8)
        Vr, _ = r.evolve(V0, np.zeros(4), s["t_reid"], m)
        self.assertAlmostEqual(r.reid(Vr, HB)["ratio"], 1.0, places=8)
        # closed form for the Reid loss time: tau^2 = u, u^2 + (c^2+2s^2)u - s^2 = 0
        c, sn = np.cosh(2 * rr), np.sinh(2 * rr)
        u = (-(c**2 + 2 * sn**2) + np.sqrt((c**2 + 2 * sn**2) ** 2 + 4 * sn**2)) / 2
        self.assertAlmostEqual(np.sqrt(u) * 2 * m * sigma0**2 / HB / s["t_reid"], 1.0, places=8)
        # scaling t2 ∝ m sigma0^2 e^{-2r}
        self.assertAlmostEqual(r.doubling_time(rr, 2 * sigma0, 3 * m) / s["t2"], 12.0, places=10)
        self.assertAlmostEqual(r.doubling_time(rr, s["sigma0_1ns"], m) / 1e-9, 1.0, places=10)

    # ---------------------------------------------------------- claim 7
    def test_claim7_spin_times_motion(self):
        for rr in (0.0, 0.5, 1.0, 2.0):
            st = r.make_pair(rr, 10e-9, HB, 2.0, ME)
            self.assertAlmostEqual(abs(r.chsh_of(st)), qm.TSIRELSON, places=12)
            reid0 = r.reid(st.cov, HB)["ratio"]
            for which in ("A", "B"):
                st2, p = r.project_spin(st, qm.direction(0.7), +1, which)
                self.assertAlmostEqual(p, 0.5)
                np.testing.assert_array_equal(st2.cov, st.cov)
                np.testing.assert_array_equal(st2.mean, st.mean)
                self.assertEqual(r.reid(st2.cov, HB)["ratio"], reid0)
                self.assertAlmostEqual(qm.concurrence(st2.rho_spin), 0.0, places=6)
        self.assertTrue(self.s["cov_unchanged_by_spin_projection"])

    # ---------------------------------------------------------- claim 8
    def test_claim8_ledger(self):
        L = self.s["ledger"]
        self.assertEqual(L["L-C3a"]["m6"], "QM")
        self.assertLessEqual(abs(L["L-C3a"]["E"]), 1.0)
        self.assertEqual(L["L-C3b"]["m6"], "QM")
        self.assertGreater(L["L-C3b"]["S"], L["L-C3b"]["classical_bound"])
        self.assertLessEqual(L["L-C3b"]["S"], L["L-C3b"]["qm_bound"])
        self.assertLess(L["L-C3c"]["epr"], L["L-C3c"]["bound"])
        self.assertAlmostEqual(L["L-C3c"]["r_eff_illustrative"], 0.294, places=3)
        self.assertAlmostEqual(1 / np.cosh(2 * 0.2944821152164407) ** 2, 0.72, places=10)
        self.assertEqual(L["L-C3d"]["grade"], "V2")


if __name__ == "__main__":
    unittest.main()


class TestTMSNegativity(unittest.TestCase):
    """Sec. 3.7.4: the unnormalized negativity (e^{2r} - 1)/2 of the
    two-mode squeezed motional pair (round-5 H16)."""

    def test_negativities(self):
        for r_, expect in ((0.5, 0.86), (1.0, 3.19), (2.0, 26.8)):
            self.assertAlmostEqual(r.tms_negativity(r_), (np.exp(2 * r_) - 1) / 2, places=6)
            self.assertAlmostEqual(r.tms_negativity(r_), expect, delta=0.05)
