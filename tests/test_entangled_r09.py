import unittest
import numpy as np
from mapping_spaces.entangled import qm, r09_vs_collapse_models as r


class R09(unittest.TestCase):
    # claim 1 — the qD model has no stochastic term: every anomaly is 0
    def test_claim1_qd_predicts_zero(self):
        for lam in (0.0, 1e-16, 2e-10, 1e-8, 1.0):
            for r_c in (1e-9, 1e-7, 1e-5):
                for m in (r.M_E, r.M_NUCLEON, 1e-12):
                    self.assertEqual(r.qd_momentum_diffusion(lam, r_c, m), 0.0)
                    self.assertEqual(r.qd_heating_rate(lam, r_c, m), 0.0)
                    self.assertEqual(r.qd_radiation_rate(lam, r_c, m), 0.0)

    # claim 2 — CSL point-particle formulae, their consistency, and the sphere form factor
    def test_claim2_csl_formulae_consistent(self):
        for m in (r.M_E, r.M_NUCLEON, 1e-12):
            Dp = r.csl_momentum_diffusion(2e-10, 1e-7, m)
            self.assertAlmostEqual(r.csl_heating_rate(2e-10, 1e-7, m) / (3 * Dp / (2 * m)), 1.0, places=12)
            # Carlesso Eq. (4): 3-D position spread = sum_i D_p t^3 / (3 m^2)
            t = 1.0
            self.assertAlmostEqual(r.csl_position_spread(2e-10, 1e-7, t) / (3 * Dp * t**3 / (3 * m**2)), 1.0, places=12)
        ex = r.csl_examples()
        self.assertAlmostEqual(ex["electron"]["D_p"] / 3.35e-71, 1.0, delta=0.01)
        self.assertAlmostEqual(ex["electron"]["dEdt_eV_s"] / 3.44e-22, 1.0, delta=0.01)
        self.assertAlmostEqual(ex["nucleon"]["dEdt_eV_s"] / 6.32e-19, 1.0, delta=0.01)
        self.assertAlmostEqual(ex["test_mass_point"]["dEdt_eV_s"] / 3.78e-4, 1.0, delta=0.01)
        # heating scales with m, diffusion with m^2
        self.assertAlmostEqual(ex["nucleon"]["dEdt_W"] / ex["electron"]["dEdt_W"], r.M_NUCLEON / r.M_E, places=9)
        self.assertAlmostEqual(ex["nucleon"]["D_p"] / ex["electron"]["D_p"], (r.M_NUCLEON / r.M_E) ** 2, places=6)

    def test_claim2_sphere_form_factor_limits(self):
        self.assertAlmostEqual(r.sphere_form_factor(1e-3 * 1e-7, 1e-7), 1.0, places=5)   # R << r_C: point
        for ratio in (20.0, 31.28, 50.0):
            ff = r.sphere_form_factor(ratio * 1e-7, 1e-7)
            self.assertAlmostEqual(ff / (6.0 / ratio**4), 1.0, delta=0.05)               # R >> r_C: 6 (r_C/R)^4
        ex = r.csl_examples()
        self.assertAlmostEqual(ex["test_mass_sphere"]["R_over_rC"], 31.28, delta=0.01)
        self.assertLess(ex["test_mass_sphere"]["form_factor"], 1e-5)
        self.assertAlmostEqual(ex["test_mass_sphere"]["D_p"] / 2.5e-40, 1.0, delta=0.02)

    # claim 3 — the free-electron table
    def test_claim3_electron_table(self):
        rows = r.electron_table()
        self.assertEqual([row["lambda"] for row in rows], [1e-16, 1e-10, 1e-8])
        for row, Dp, dE, t in zip(rows, (1.67e-77, 1.67e-71, 1.67e-69),
                                  (1.72e-28, 1.72e-22, 1.72e-20), (4.96e18, 4.96e12, 4.96e10)):
            self.assertAlmostEqual(row["D_p"] / Dp, 1.0, delta=0.01)
            self.assertAlmostEqual(row["dEdt_eV_s"] / dE, 1.0, delta=0.01)
            self.assertAlmostEqual(row["t_to_fraction_s"] / t, 1.0, delta=0.01)
        # even Adler's value needs > 1000 yr to shift an electron's momentum spread by 1e-3 p_0
        self.assertGreater(rows[-1]["t_to_fraction_yr"], 1e3)

    # claim 4 — CSL acts on positions only: spin CHSH factorises
    def test_claim4_spin_factorisation(self):
        for lam, ex in ((r.LAMBDA_GRW, 1.0), (r.LAMBDA_CANTILEVER, 1.0), (r.LAMBDA_CANTILEVER, 1e17), (1e-8, 1e30)):
            out = r.factorisation_check(lam=lam, exaggerate=ex)
            self.assertAlmostEqual(out["product"]["S_before"], qm.TSIRELSON, places=12)
            self.assertAlmostEqual(out["product"]["S_after"], qm.TSIRELSON, places=12)
            self.assertAlmostEqual(out["product"]["concurrence_after"], 1.0, places=9)
        # the discrete model reproduces the two-particle rate: twice Gamma(d) of one electron at d = 1 um >> r_C
        out = r.factorisation_check()
        self.assertAlmostEqual(out["two_particle_rate_s"] / (2 * out["single_particle_rate_s"]), 1.0, places=9)
        self.assertAlmostEqual(out["single_particle_rate_s"] / (r.LAMBDA_CANTILEVER * (r.M_E / r.AMU) ** 2), 1.0, places=9)
        self.assertAlmostEqual(out["two_particle_rate_s"] / 1.20e-16, 1.0, delta=0.01)

    def test_claim4_interferometric_storage_is_the_only_handle(self):
        # spin stored in a path superposition for one year at the cantilever bound: 5e-9 relative loss
        out = r.factorisation_check()
        self.assertAlmostEqual(out["interferometric"]["S_before"], qm.TSIRELSON, places=12)
        self.assertLess(out["interferometric"]["S_after"], qm.TSIRELSON)
        self.assertLess(qm.TSIRELSON - out["interferometric"]["S_after"], 1e-7)
        self.assertAlmostEqual(out["gamma_t_max"], 3.8e-9, delta=0.05e-9)
        # fully decohered limit: the recombined spin state is a classical mixture, S_max = 2
        out = r.factorisation_check(exaggerate=1e17)
        self.assertAlmostEqual(out["interferometric"]["S_after"], 2.0, places=9)
        self.assertAlmostEqual(out["interferometric"]["concurrence_after"], 0.0, places=9)

    # claim 5 — exclusion status and the GRW target
    def test_claim5_exclusion_and_target(self):
        grw = r.exclusion_status(1e-16)
        self.assertFalse(grw["excluded"])
        mid = r.exclusion_status(1e-10)
        self.assertTrue(mid["excluded"])
        self.assertTrue(mid["allowed_by_cantilever_alone"])
        self.assertTrue(all("X-ray" in k for k in mid["excluded_by"]))
        adler = r.exclusion_status(1e-8)
        self.assertTrue(adler["excluded"])
        self.assertFalse(adler["allowed_by_cantilever_alone"])
        self.assertIn("L-C2a/L-C2d cantilever", adler["excluded_by"])
        tgt = r.grw_target()
        self.assertEqual(tgt["target_lambda"], 1e-16)
        self.assertAlmostEqual(tgt["factor_vs_xray_review"], 5200.0)
        self.assertAlmostEqual(tgt["factor_vs_xray_majorana"], 1700.0)
        self.assertAlmostEqual(tgt["factor_vs_xray_nuclear"], 49.0)
        self.assertAlmostEqual(tgt["factor_vs_cantilever"], 2.0e6)

    # claim 6 — Diósi–Penrose
    def test_claim6_diosi_penrose(self):
        dp = r.dp_examples()
        self.assertAlmostEqual(dp["nucleon_R0_1e-15_erg_s"] / 1.66e-20, 1.0, delta=0.01)
        self.assertAlmostEqual(dp["nucleon_R0_1e-15_K_s"] / 8.0e-5, 1.0, delta=0.01)   # review: ~7e-5 K/s
        self.assertAlmostEqual(dp["suppression_at_2.54e-10"] / 6.1e-17, 1.0, delta=0.01)
        self.assertAlmostEqual(dp["nucleon_R0_2.54e-10_W"] / 1.01e-43, 1.0, delta=0.01)
        self.assertEqual(dp["qd_prediction_W"], 0.0)
        # DP heating = total diffusion / 2m
        self.assertAlmostEqual(r.dp_heating_rate(r.M_NUCLEON, 1e-15) * 2 * r.M_NUCLEON
                               / r.dp_momentum_diffusion_total(r.M_NUCLEON, 1e-15), 1.0, places=12)
        # coefficient verified against Nimmrichter, Hornberger, Hammerer 2014 Eq. (11):
        # D_DP = G hbar/(2 pi^2) Int d^3k (k_x^2/k^2) |rho~(k)|^2, Gaussian of variance R0^2 -> |rho~|^2 = m^2 e^{-k^2 R0^2}
        for m, R0 in ((r.M_NUCLEON, 1e-15), (1e-12, 2.54e-10)):
            k = np.linspace(0.0, 12.0 / R0, 400001)
            I = (1.0 / 3.0) * 4 * np.pi * np.trapezoid(k**2 * np.exp(-(k**2) * R0**2), k)   # Int d^3k (k_x^2/k^2) e^{-k^2 R0^2}
            D_x = r.G_NEWTON * r.HBAR / (2 * np.pi**2) * m**2 * I
            self.assertAlmostEqual(3 * D_x / r.dp_momentum_diffusion_total(m, R0), 1.0, places=6)
            self.assertAlmostEqual(D_x / (r.G_NEWTON * r.HBAR * m**2 / (6 * np.sqrt(np.pi) * R0**3)), 1.0, places=6)
        # Donadi et al. 2021 (4 pi G / hbar normalisation) quote dT/dt = 4 sqrt(pi) m_0 G hbar / (3 k_B R0^3): 8 pi x this row
        dT_donadi = 4 * np.sqrt(np.pi) * r.M_NUCLEON * r.G_NEWTON * r.HBAR / (3 * r.K_B * 1e-45)
        self.assertAlmostEqual(dT_donadi / dp["nucleon_R0_1e-15_K_s"] / (8 * np.pi), 1.0, places=9)


if __name__ == "__main__":
    unittest.main()
