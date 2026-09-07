import unittest
import numpy as np
from mapping_spaces.entangled import qm, r07_realistic_tracking as r

H24 = r.HBAR ** 2 / 4


class R07(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rel = r.relaxation_run(1e19)                                   # mixed start, deterministic
        cls.ens = r.simulate_pair(1e19, 1e-7, n_steps=2000, n_traj=2000)   # steady start, ensemble
        cls.pure = r.simulate_pair(1e19, 3e-7, n_steps=3000, n_traj=50, start="pure", sigma0=3e-6)

    # claim 1 — steady state closed form
    def test_claim1_steady_state_closed_form(self):
        ss = r.steady_state(1e19)
        self.assertAlmostEqual(ss["C"], r.HBAR / 2)
        self.assertAlmostEqual(ss["Vx"], np.sqrt(r.HBAR / (8 * r.M_E * 1e19)))
        self.assertAlmostEqual(ss["D"] / H24, 1.0, places=12)
        rhs = r.covariance_rhs(ss["Vx"], ss["Vp"], ss["C"], 1e19)
        scale = (2 * ss["C"] / r.M_E, 2 * r.HBAR ** 2 * 1e19, ss["Vp"] / r.M_E)
        for val, sc in zip(rhs, scale):                                    # fixed point to round-off
            self.assertLess(abs(val) / sc, 1e-12)
        for key in ("Vx", "Vp", "C"):
            self.assertAlmostEqual(self.rel[key][-1] / ss[key], 1.0, places=9)
        # scaling: V_x ~ k^-1/2, V_p ~ k^+1/2
        self.assertAlmostEqual(r.steady_state(1e21)["Vx"] / ss["Vx"], 0.1, places=12)
        self.assertAlmostEqual(r.steady_state(1e21)["Vp"] / ss["Vp"], 10.0, places=9)

    # claim 2 — uncertainty product never below hbar^2/4
    def test_claim2_uncertainty_product(self):
        D, Vx, t = self.rel["D"], self.rel["Vx"], self.rel["t"]
        self.assertAlmostEqual(self.rel["D0"] / H24, 8.0)
        self.assertTrue(np.all(D >= H24 * (1 - 1e-12)))
        self.assertTrue(np.all(np.diff(D) <= 0))
        self.assertAlmostEqual(D[-1] / H24, 1.0, places=9)
        dt = t[1] - t[0]
        lhs = (D[2:] - D[:-2]) / (2 * dt)
        rhs = -8 * 1e19 * Vx[1:-1] * (D[1:-1] - H24)
        n = 500                                                            # first 5/gamma
        np.testing.assert_allclose(lhs[:n], rhs[:n], rtol=5e-3)
        # a pure packet stays minimum-uncertainty along the whole stochastic run
        self.assertTrue(np.all(self.pure["D"] >= H24 * (1 - 1e-12)))
        self.assertLess(np.abs(self.pure["D"] / H24 - 1).max(), 1e-6)

    # claim 3 — momentum diffusion 2 hbar^2 k t, heating hbar^2 k/m
    def test_claim3_momentum_diffusion(self):
        md = r.momentum_diffusion(self.ens)
        np.testing.assert_allclose(md["identity"], md["expected_identity"], rtol=1e-12)
        self.assertAlmostEqual(md["ensemble"][-1] / md["theory"][-1], 1.0, delta=0.03)
        self.assertAlmostEqual(r.heating_rate(1e19) / r.EV, 0.762, delta=0.002)
        self.assertAlmostEqual(r.heating_rate(1e23) / r.EV, 7.62e3, delta=20)

    # claim 4 — precision of the fitted separation speed 2/a
    def test_claim4_fit_precision(self):
        slopes = r.fitted_separation_speed(self.ens)
        self.assertAlmostEqual(slopes.mean() / (2 * r.V0), 1.0, delta=0.001)
        cf = np.sqrt(r.slope_variance_closed_form(1e19, 1e-7))
        self.assertAlmostEqual(slopes.std() / cf, 1.0, delta=0.05)
        self.assertAlmostEqual(cf / (2 * r.V0), 0.0092, places=4)
        rec = r.fitted_separation_speed(self.ens, use_record=True)
        cf_rec = np.sqrt(r.slope_variance_closed_form(1e19, 1e-7, record=True))
        self.assertAlmostEqual(rec.std() / cf_rec, 1.0, delta=0.05)
        self.assertLess(cf_rec, cf)                                        # the smoother beats the filter
        grid = r.precision_grid()
        np.testing.assert_allclose(grid[0], [0.0101, 0.0092, 0.0229], rtol=0.02)
        np.testing.assert_allclose(grid[2], [0.229, 0.708, 2.23], rtol=0.02)
        opt = r.optimal_run_time(1e19)
        self.assertAlmostEqual(opt["T_opt"], 3.74e-8, delta=0.02e-8)
        self.assertAlmostEqual(opt["rel_err_min"], 0.0081, places=4)

    # claim 5 — frames attached to the tracks: zero-mean separation in A's frame
    def test_claim5_frames_on_tracked_paths(self):
        sf = r.separation_in_frames(self.ens, traj=0)
        direct = (self.ens["xB"] - self.ens["xA"])[0]
        np.testing.assert_allclose(sf["lab"], direct, rtol=1e-12, atol=1e-18)
        np.testing.assert_allclose(sf["A"], direct - 2 * self.ens["t"] / r.A_PARAM, rtol=1e-9, atol=1e-18)
        np.testing.assert_allclose(sf["A"], sf["B"], rtol=1e-12, atol=1e-18)
        self.assertAlmostEqual(sf["lab"][-1], 2e-7 / r.A_PARAM, delta=0.02 * 2e-7 / r.A_PARAM)
        sepA = r.separation_in_A_frame_ensemble(self.ens)[:, -1]
        self.assertLess(abs(sepA.mean()) / sepA.std(), 0.1)
        self.assertAlmostEqual(sepA.std() / np.sqrt(r.separation_variance_closed_form(1e19, 1e-7)), 1.0, delta=0.05)
        self.assertAlmostEqual(sepA.std(), 1.77e-5, delta=0.1e-5)

    # claim 6 — spin untouched
    def test_claim6_spin_unchanged(self):
        out = r.spin_chsh_before_after()
        self.assertAlmostEqual(abs(out["S_before"]), qm.TSIRELSON, places=12)
        self.assertEqual(out["S_before"], out["S_after"])
        self.assertAlmostEqual(out["concurrence"], 1.0, places=9)
        pos = r.reduced_spin_change(coupling="position")
        self.assertLess(pos["norm_change"], 1e-12)
        self.assertAlmostEqual(abs(pos["S_after_step"]), qm.TSIRELSON, places=12)
        grad = r.reduced_spin_change(coupling="gradient")
        self.assertGreater(grad["norm_change"], 1.0)
        self.assertLess(abs(grad["S_after_step"]), qm.TSIRELSON - 0.1)
        self.assertGreater(abs(grad["S_after_step"]), 2.0)
        self.assertAlmostEqual(abs(grad["S_after_step"]), 2.53, delta=0.01)  # the seeded instance
        self.assertAlmostEqual(grad["x2"], 1.066, delta=0.001)
        self.assertAlmostEqual(abs(grad["S_after_step"]), grad["S_closed_form"], places=9)
        # round-5 H3: the closed form sqrt2 (2 - 0.2 <x^2>) and the uniform-grid instance
        uni = r.reduced_spin_change(coupling="gradient", position_state="uniform")
        self.assertAlmostEqual(uni["x2"], 1.25, places=12)
        self.assertAlmostEqual(abs(uni["S_after_step"]), 2.475, places=3)
        self.assertAlmostEqual(abs(uni["S_after_step"]), np.sqrt(2) * (2 - 0.2 * 1.25), places=9)
        self.assertAlmostEqual(r.spin_step_closed_form(1.25), 2.475, places=3)
        self.assertAlmostEqual(r.spin_dephasing_rate_ratio(), 4.0, places=6)  # G1: 4 Gamma <x^2>

    # claim 7 — realistic numbers
    def test_claim7_table(self):
        rows = {row["k"]: row for row in r.realistic_table()}
        self.assertAlmostEqual(rows[1e19]["sqrtVx_m"], 1.10e-6, delta=0.01e-6)
        self.assertAlmostEqual(rows[1e23]["sqrtVx_m"], 110e-9, delta=1e-9)
        self.assertAlmostEqual(rows[1e27]["sqrtVx_m"], 11.0e-9, delta=0.1e-9)
        self.assertAlmostEqual(rows[1e19]["gamma"], 9.62e7, delta=0.01e7)
        self.assertAlmostEqual(rows[1e19]["t_double_KE_s"], 3.73e-4, delta=0.01e-4)
        self.assertAlmostEqual(rows[1e21]["heating_eV_per_s"], 76.2, delta=0.2)
        self.assertAlmostEqual(rows[1e23]["rel_err_2_over_a"], 0.708, places=2)


if __name__ == "__main__":
    unittest.main()
