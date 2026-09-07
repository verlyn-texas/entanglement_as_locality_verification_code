import unittest

import numpy as np

from mapping_spaces.entangled import graph_frames as G, qm, r12_swapping as r

YS = (0.5, 1.25, 1.75, 2.5, 3.25, 4.0)
A_SET, D_SET = qm.direction(0.4, 0.2), qm.direction(1.9, -0.6)


class R12(unittest.TestCase):
    def test_claim1_geometry_and_transfer(self):
        self.assertEqual(r.meeting_point(), (1.0, 2.0))
        h = r.history("a")
        tl = r.locality_timeline(h, YS)
        for y in (0.5, 1.25, 1.75):
            self.assertEqual(tl[y], {"AB": True, "CD": True, "AD": False})
        self.assertEqual(tl[2.5], {"AB": False, "CD": False, "AD": True})  # monogamy in the graph
        # M10': no transfer
        tl2 = r.locality_timeline(r.history("a", transfer=False), YS)
        self.assertEqual(tl2[2.5], {"AB": False, "CD": False, "AD": False})
        # bell_measure refuses a non-co-located pair
        with self.assertRaises(ValueError):
            G.History(r.swap_graph()).bell_measure("B", "C", 1.0)
        # continuity through the BSM in every frame: jump bounded by 2 eps * max speed (2)
        self.assertLess(r.continuity_jumps(h, eps=1e-3), 4e-3)
        # A-D separation: lab 2 + 2y; A and D frames constant (6) on (Y_BSM, y_A)
        sep = r.ad_separations(h, (0.0, 1.0, 2.0, 2.5, 3.0, 4.0))
        np.testing.assert_allclose(sep[G.LAB], [2, 4, 6, 7, 8, 10], atol=1e-12)
        np.testing.assert_allclose(sep["A"], [2, 4, 6, 6, 6, 8], atol=1e-12)
        np.testing.assert_allclose(sep["D"], [2, 4, 6, 6, 6, 8], atol=1e-12)

    def test_claim2_bsm_statistics(self):
        a, ap, b, bp = qm.chsh_optimal_settings()
        for row in r.bsm_analysis():
            self.assertAlmostEqual(row["prob"], 0.25, places=12)
            self.assertAlmostEqual(row["F_bell"], 1.0, places=12)
            self.assertAlmostEqual(row["F_corrected"], 1.0, places=12)
            self.assertAlmostEqual(row["chsh_corrected"], qm.TSIRELSON, places=12)
            self.assertAlmostEqual(row["concurrence"], 1.0, places=9)
        # averaged (uncorrected) state is 1/4: CHSH 0
        np.testing.assert_allclose(r.averaged_state(), np.eye(4) / 4, atol=1e-12)
        self.assertAlmostEqual(qm.chsh(r.averaged_state(), a, ap, b, bp), 0.0, places=12)
        self.assertAlmostEqual(sum(x["prob"] * x["chsh_uncorrected"] for x in r.bsm_analysis()), 0.0, places=12)
        # no-signalling: A's and D's marginals do not depend on Victor's choice
        marg = r.no_signalling_marginals()
        for choice in ("none", "bsm", "ssm"):
            for w in ("A", "D"):
                self.assertAlmostEqual(marg[choice][w], 0.5, places=12)
        # Bell-state labels: (sigma_k x 1)|psi-> is psi-, phi-, phi+, psi+ up to phase
        phi_p, phi_m, psi_p = qm.ket(1, 0, 0, 1), qm.ket(1, 0, 0, -1), qm.ket(0, 1, 1, 0)
        for k, target in ((1, phi_m), (2, phi_p), (3, psi_p)):
            self.assertAlmostEqual(abs(target.conj() @ r.bell_state(k)), 1.0, places=12)

    def test_claim3_orderings_give_the_same_joint_distribution(self):
        Ja, Jb = r.joint_case_a(A_SET, D_SET), r.joint_case_b(A_SET, D_SET)
        self.assertAlmostEqual(Ja.sum(), 1.0, places=12)
        np.testing.assert_allclose(Ja, Jb, atol=1e-12)
        self.assertEqual(r.commutator_norm(A_SET, D_SET), 0.0)
        for k in range(4):
            self.assertAlmostEqual(r.conditional_correlation(Ja, k), r.conditional_correlation(Jb, k), places=12)
            self.assertAlmostEqual(r.post_selected_chsh(r.joint_case_a, k), qm.TSIRELSON, places=12)
            self.assertAlmostEqual(r.post_selected_chsh(r.joint_case_b, k), qm.TSIRELSON, places=12)
        # outcome psi-: the conditional correlation is the singlet's -a.d
        self.assertAlmostEqual(r.conditional_correlation(Ja, 0), -float(A_SET @ D_SET), places=12)

    def test_claim4_case_b_narrative_reproduces_case_a(self):
        Ja = r.joint_case_a(A_SET, D_SET)
        nar = r.narrative_case_b(A_SET, D_SET)
        np.testing.assert_allclose(nar["joint"], Ja, atol=1e-12)
        for s in (+1, -1):
            self.assertAlmostEqual(nar["checks"][s]["F_B(-s a)"], 1.0, places=12)  # A changes B ...
            self.assertAlmostEqual(nar["checks"][s]["D_max_dev_from_I/2"], 0.0, places=12)  # ... not D
        for k in range(4):
            self.assertAlmostEqual(r.post_selected_chsh(lambda x, y: r.narrative_case_b(x, y)["joint"], k),
                                   qm.TSIRELSON, places=12)
        # case (a): after the BSM and correction, A's projection leaves D pure in |-s a>
        for F in r.narrative_case_a(A_SET, D_SET).values():
            self.assertAlmostEqual(F, 1.0, places=12)

    def test_claim5_ssm_separable_witness_sign(self):
        for row in r.ssm_analysis():
            self.assertAlmostEqual(row["prob"], 0.25, places=12)
            self.assertAlmostEqual(row["concurrence"], 0.0, places=9)
            self.assertAlmostEqual(row["F_best"], 0.5, places=12)
            self.assertAlmostEqual(row["W"], 0.0, places=12)
        for W in r.bsm_witnesses():
            self.assertAlmostEqual(W, -0.5, places=12)
        # ledger L-D2 sign pattern: BSM witness negative, SSM non-negative
        self.assertLess(-0.181, 0.0)
        self.assertGreaterEqual(0.078, 0.0)

    def test_claim6_case_b_graph_never_links_a_and_d(self):
        h = r.history("b")
        tl = r.locality_timeline(h, YS)
        self.assertEqual(tl[0.5], {"AB": True, "CD": True, "AD": False})
        self.assertEqual(tl[1.25], {"AB": False, "CD": True, "AD": False})  # A projected at y = 1
        self.assertEqual(tl[1.75], {"AB": False, "CD": False, "AD": False})  # D projected at y = 1.5
        for y in np.linspace(0.0, 5.0, 101):
            self.assertFalse(h.graph_at(y).is_local("A", "D"))
        # at the BSM, B and C have no partners left, so nothing is transferred
        g = h.graph_at(r.Y_BSM - 1e-9)
        self.assertEqual(g.partners("B"), [])
        self.assertEqual(g.partners("C"), [])
        # case (a): A~D local exactly on (Y_BSM, y_A)
        ha = r.history("a")
        self.assertTrue(ha.graph_at(2.5).is_local("A", "D"))
        self.assertFalse(ha.graph_at(3.5).is_local("A", "D"))

    def test_claim7_m10_vs_m10prime(self):
        with_t = r.graph_vs_state("a", True, YS)
        without = r.graph_vs_state("a", False, YS)
        for u, v in zip(with_t, without):
            self.assertEqual(u["concurrence_AD"], v["concurrence_AD"])  # same statistics
            if u["y"] != 2.5:
                self.assertEqual(u["local_AD"], v["local_AD"])
        self.assertTrue(with_t[3]["local_AD"] and with_t[3]["concurrence_AD"] > 0.999)
        self.assertFalse(without[3]["local_AD"])
        self.assertGreater(without[3]["concurrence_AD"], 0.999)  # entangled yet not local

    def test_claim8_heights_and_interpolant_degree(self):
        g = r.swap_graph()
        self.assertEqual(len({p.z for p in g.particles.values()}), 4)
        self.assertEqual(len({p.v for p in g.particles.values()}), 4)
        for k in ("A", "B", "C", "D", G.LAB):
            self.assertEqual(r.interpolant_degree(g, k), 4)
        # pinned values: lab sees each at its velocity; A sees D at rest only when linked
        for n in r.NAMES:
            self.assertAlmostEqual(g.velocity_in_frame(n, G.LAB), r.VELOCITIES[n])
        self.assertAlmostEqual(g.velocity_in_frame("D", "A"), 2.0)
        self.assertAlmostEqual(g.link("A", "D").velocity_in_frame("D", "A"), 0.0)
        # a coincident height with a different velocity is refused (1-D qD needs distinct heights)
        with self.assertRaises(ValueError):
            G.LocalityGraph().add(G.Particle("A", -1.0, -1.0)).add(G.Particle("E", -1.0, 0.3))

    def test_claim9_werner_visibility_multiplies(self):
        for p1, p2 in ((0.9, 0.8), (1.0, 0.65), (0.7, 0.7)):
            out = r.werner_swap(p1, p2)
            for k in range(4):
                self.assertAlmostEqual(out[k]["prob"], 0.25, places=12)
                self.assertAlmostEqual(out[k]["dev_from_werner"], 0.0, places=12)
                self.assertAlmostEqual(out[k]["visibility"], p1 * p2, places=12)
                self.assertAlmostEqual(out[k]["chsh"], qm.TSIRELSON * p1 * p2, places=12)
        # ledger-derived visibilities (L-D3, L-D7) and L-D1's 0.65 as a product
        self.assertAlmostEqual(r.visibility_from_chsh(2.421), 0.856, places=3)
        self.assertAlmostEqual(r.visibility_from_chsh(2.37), 0.838, places=3)
        self.assertAlmostEqual(r.visibility_from_chsh(2.38), 0.841, places=3)
        self.assertAlmostEqual(np.sqrt(0.65), 0.806, places=3)
        self.assertLess(qm.TSIRELSON * 0.65, 2.0)  # L-D1: no Bell violation at visibility 0.65


if __name__ == "__main__":
    unittest.main()


class TestWhenRule(unittest.TestCase):
    """Round-5 T1: the charts are read off the outcome-conditioned state;
    single-particle projections agree with the dephased reading, a Bell-basis
    projection on two partners does not (the swapped edge jumps 0 -> 1)."""

    def test_dephased_vs_conditional(self):
        out = r.dephased_vs_conditional_weights()
        self.assertAlmostEqual(out["singlet_Z"]["dephased"], 0.0, places=9)
        for c in out["singlet_Z"]["conditional"]:
            self.assertAlmostEqual(c, 0.0, places=9)
        self.assertAlmostEqual(out["ghz_X"]["dephased"], 1.0, places=9)
        for c in out["ghz_X"]["conditional"]:
            self.assertAlmostEqual(c, 1.0, places=9)
        sw = out["swap_BSM"]
        self.assertAlmostEqual(sw["dephased"]["AD"], 0.0, places=9)
        self.assertAlmostEqual(sw["dephased"]["BC"], 0.0, places=9)
        for c in sw["conditional"]:
            self.assertAlmostEqual(c["AD"], 1.0, places=9)
            self.assertAlmostEqual(c["BC"], 1.0, places=9)
