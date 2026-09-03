import itertools
import unittest

import numpy as np

from mapping_spaces.entangled import graph_frames as G, qm
from mapping_spaces.entangled import r13_ghz as r13
from mapping_spaces.entangled import r17_mixed_multipartite as r


class R17(unittest.TestCase):
    def test_claim1_tools_and_pure_state_reduction(self):
        # partial transpose: involution, trace- and Hermiticity-preserving
        rho = r.white_noise(r.w_state(3), 0.7)
        for S in r.cuts(3):
            pt = r.partial_transpose(rho, S, 3)
            np.testing.assert_allclose(r.partial_transpose(pt, S, 3), rho, atol=1e-12)
            self.assertAlmostEqual(np.trace(pt).real, 1.0, places=12)
            np.testing.assert_allclose(pt, pt.conj().T, atol=1e-12)
        self.assertEqual(len(list(r.cuts(3))), 3)
        self.assertEqual(len(list(r.cuts(4))), 7)
        # two qubits: PPT <=> C = 0 (Peres-Horodecki, L-E5c) and 2N <= C
        chk = r.two_qubit_checks(150, 17)
        self.assertEqual(chk["ppt_iff_separable"], chk["states"])
        self.assertEqual(chk["2N_le_C"], chk["states"])
        self.assertGreater(chk["entangled"], 200)
        # pure states: negativities and the reduction to R13's components
        for S in r.cuts(3):
            self.assertAlmostEqual(r.negativity(qm.dm(r.ghz(3)), S, 3), 0.5, places=12)
        for k in range(3):
            self.assertAlmostEqual(r.negativity(qm.dm(r.w_state(3)), (k,), 3), np.sqrt(2) / 3, places=12)
        for lab, st, n, r13_comps in r.pure_state_cases():
            comps, transitive = r.components(st, n)
            self.assertEqual(comps, r13_comps, lab)
            self.assertTrue(transitive, lab)
        # the "some separating cut NPT" reading merges two independent Bell pairs
        bb = np.kron(qm.singlet(), qm.singlet())
        self.assertEqual(r.components(bb, 4)[0], [(0, 1), (2, 3)])
        self.assertEqual(r.components(bb, 4, relation=r.same_component_naive)[0], [(0, 1, 2, 3)])
        # weights: pure GHZ_N -> 1 (single-particle cuts); the all-cuts variant gives 1/3 for GHZ_4
        self.assertAlmostEqual(r.component_weight(qm.dm(r.ghz(3)), (0, 1, 2), 3), 1.0, places=9)
        self.assertAlmostEqual(r.component_weight(qm.dm(r.ghz(4)), (0, 1, 2, 3), 4), 1.0, places=9)
        self.assertAlmostEqual(r.weakest_cut_all(qm.dm(r.ghz(4)), (0, 1, 2, 3), 4), 1 / 3, places=9)
        self.assertAlmostEqual(r.component_weight(qm.dm(r.w_state(3)), (0, 1, 2), 3), 2 * np.sqrt(2) / 3, places=9)
        # a two-qubit component carries the concurrence (R05)
        self.assertAlmostEqual(r.component_weight(np.kron(qm.werner(0.8), qm.projector(qm.direction(0.0), 1)), (0, 1), 3), 0.7, places=9)

    def test_claim2_noisy_ghz(self):
        g = r13.triple()
        ps = np.linspace(0.0, 1.0, 21)
        for p in ps:
            negs = r.cut_negativities(r.ghz(3), p, 3)
            for v in negs.values():
                self.assertAlmostEqual(v, r.ghz_negativity_closed(p, 3), places=12)
            for c in r.pairwise_concurrences(r.ghz(3), p).values():
                self.assertAlmostEqual(c, 0.0, places=9)
            rho = r.white_noise(r.ghz(3), p)
            comps, transitive = r.components(rho, 3)
            self.assertTrue(transitive)
            if p <= 0.2 + 1e-12:
                self.assertEqual(comps, [(0,), (1,), (2,)])
                self.assertEqual(r.edges(r.locality_graph(rho, g)), [])
            else:
                self.assertEqual(comps, [(0, 1, 2)])
                w = r.edge_weights(r.locality_graph(rho, g))
                self.assertEqual(sorted(w), [("P1", "P2"), ("P1", "P3"), ("P2", "P3")])
                for val in w.values():
                    self.assertAlmostEqual(val, r.ghz_component_weight_closed(p), places=9)
            self.assertEqual(r.edges(r.pairwise_graph(rho, g)), [])
            # the graph feeds the frames: locality fraction = weight (R11 claim 5)
            if p > 0.2 + 1e-12:
                self.assertAlmostEqual(r.locality_graph(rho, g).locality_fraction("P1", "P2"), r.ghz_component_weight_closed(p), places=9)
        # thresholds by bisection
        self.assertAlmostEqual(r.ppt_threshold(r.ghz(3), 3), 0.2, places=8)
        self.assertAlmostEqual(r.ghz_ppt_threshold_formula(3), 0.2, places=12)
        self.assertAlmostEqual(r.biseparable_overlap(r.ghz(3), 3), 0.5, places=12)
        self.assertAlmostEqual(r.witness_threshold(r.ghz(3), 3), 3 / 7, places=8)
        self.assertAlmostEqual(r.ghz_witness_threshold_formula(3), 3 / 7, places=12)
        self.assertAlmostEqual(r.mermin_threshold(), 0.5, places=8)
        # the window: one component, witness not negative
        for p in (0.25, 0.3, 3 / 7):
            rho = r.white_noise(r.ghz(3), p)
            self.assertEqual(r.components(rho, 3)[0], [(0, 1, 2)])
            self.assertGreaterEqual(r.projector_witness(r.ghz(3), rho, 3), -1e-12)
        self.assertAlmostEqual(r.projector_witness(r.ghz(3), r.white_noise(r.ghz(3), 0.5), 3), -0.0625, places=12)
        self.assertAlmostEqual(r.ghz_component_weight_closed(3 / 7), 2 / 7, places=12)

    def test_claim3_noisy_w(self):
        g = r13.triple()
        w3 = r.w_state(3)
        self.assertAlmostEqual(r.pairwise_concurrences(w3, 1.0)[(0, 1)], 2 / 3, places=9)
        p_c = r.w_pair_threshold()
        p_n = r.ppt_threshold(w3, 3)
        self.assertAlmostEqual(p_c, r.w_pair_threshold_formula(), places=8)
        self.assertAlmostEqual(p_c, 0.5482, places=4)
        self.assertAlmostEqual(p_n, r.w_cut_threshold_formula(), places=8)
        self.assertAlmostEqual(p_n, 0.2096, places=4)
        self.assertLess(p_n, p_c)  # cuts go PPT *before* (at lower p than) the pairwise edges vanish
        # all three cuts share the threshold
        for k in range(3):
            self.assertAlmostEqual(r.ppt_threshold(w3, 3, (k,)), p_n, places=8)
        # witness: alpha = 2/3, threshold 13/21 > p_C
        self.assertAlmostEqual(r.biseparable_overlap(w3, 3), 2 / 3, places=12)
        self.assertAlmostEqual(r.witness_threshold(w3, 3), 13 / 21, places=8)
        self.assertGreater(13 / 21, p_c)
        # the window (p_N, p_C]: component rule one component, pairwise rule no edge
        for p in (0.25, 0.4, 0.54):
            rho = r.white_noise(w3, p)
            self.assertEqual(r.components(rho, 3)[0], [(0, 1, 2)])
            self.assertEqual(r.edges(r.pairwise_graph(rho, g)), [])
            self.assertEqual(len(r.edges(r.locality_graph(rho, g))), 3)
        rho = r.white_noise(w3, 0.15)
        self.assertEqual(r.components(rho, 3)[0], [(0,), (1,), (2,)])
        # above p_C both rules give three edges but different weights; closed forms
        for p in (0.6, 0.8, 1.0):
            rho = r.white_noise(w3, p)
            wc = r.edge_weights(r.locality_graph(rho, g))[("P1", "P2")]
            wp = r.edge_weights(r.pairwise_graph(rho, g))[("P1", "P2")]
            self.assertAlmostEqual(wc, 2 * np.sqrt(2) / 3 * p - (1 - p) / 4, places=9)
            self.assertAlmostEqual(wp, 2 * max(0.0, p / 3 - np.sqrt((p / 3 + (1 - p) / 4) * (1 - p) / 4)), places=9)
            self.assertGreater(wc, wp)
        self.assertAlmostEqual(r.edge_weights(r.pairwise_graph(r.white_noise(w3, 0.6), g))[("P1", "P2")], 0.0536, places=4)

    def test_claim4_dephasing_and_frames(self):
        ts = np.linspace(0.0, 2.0, 9)
        gamma = 1.0
        for n in (2, 3, 4):
            tr = r.dephasing_trace(r.ghz(n), n, gamma, ts)
            np.testing.assert_allclose(tr["negativity"], r.ghz_dephasing_closed(n, gamma, ts), atol=1e-12)
            self.assertFalse(tr["ppt"].any())  # never PPT at finite t
            np.testing.assert_allclose(tr["weight"], np.exp(-n * gamma * ts), atol=1e-9)
            if n == 2:
                np.testing.assert_allclose(tr["concurrence"], np.exp(-2 * gamma * ts), atol=1e-9)
            else:
                np.testing.assert_allclose(tr["concurrence"], 0.0, atol=1e-9)
            self.assertAlmostEqual(r.lifetime(n, gamma), 1 / n)
            self.assertAlmostEqual(r.time_to_weight(n, gamma, 0.01), np.log(100) / n, places=12)
        # W_3: coherences decay at 2 gamma, N-independent
        tw = r.dephasing_trace(r.w_state(3), 3, gamma, ts)
        neg_c, conc_c = r.w3_dephasing_closed(gamma, ts)
        np.testing.assert_allclose(tw["negativity"], neg_c, atol=1e-12)
        np.testing.assert_allclose(tw["concurrence"], conc_c, atol=1e-9)
        self.assertFalse(tw["ppt"].any())
        # for N >= 3 the W component outlives the GHZ component
        self.assertGreater(tw["weight"][-1], r.dephasing_trace(r.ghz(3), 3, gamma, ts)["weight"][-1])
        # local depolarising: finite death, weakest cut is a single-particle cut
        deaths = {n: r.component_death_depolarising(r.ghz(n), n) for n in (2, 3, 4)}
        self.assertAlmostEqual(deaths[2][0], 1 - 1 / np.sqrt(3), places=8)
        self.assertAlmostEqual(deaths[4][0], 1 - 1 / np.sqrt(3), places=8)
        self.assertAlmostEqual(deaths[3][0], 0.4433, places=4)
        self.assertEqual(deaths[4][1], (0,))
        self.assertAlmostEqual(r.depolarising_death_time(deaths[3][0], 1.0), 0.5857, places=4)
        # frame kinematics: separation in a partner's frame lags the lab track by (v_j - v_i)/(N gamma)
        y = np.linspace(0.0, 3.0, 3001)
        v1, v2, v3 = r13.VELOCITIES
        sep = r.separation_under_decay(gamma, y, "P1", "P2")
        np.testing.assert_allclose(sep, r.separation_closed(gamma, y, v1, v2), atol=1e-5)
        self.assertAlmostEqual(r.frame_lag(gamma, v1, v2), 2 / 3, places=12)
        lab_sep = (v2 - v1) * y
        self.assertAlmostEqual(lab_sep[-1] - sep[-1], r.frame_lag(gamma, v1, v2), places=3)
        sep23 = r.separation_under_decay(gamma, y, "P2", "P3")
        np.testing.assert_allclose(sep23, r.separation_closed(gamma, y, v2, v3), atol=1e-5)
        # sanity: lab tracks are untouched by the weights
        g = r13.triple()
        self.assertAlmostEqual(g.velocity_in_frame("P2", G.LAB), v2, places=12)

    def test_claim5_ledger_mapping(self):
        rows = r.ledger_table(bisect_up_to=6)
        by = {(row["entry"], row["N"]): row for row in rows}
        self.assertAlmostEqual(by[("L-D6b", 2)]["p"], 0.9813, places=4)
        self.assertAlmostEqual(by[("L-D6b", 3)]["p"], 0.9657, places=4)
        self.assertAlmostEqual(by[("L-D6b", 4)]["p"], 0.9541, places=4)
        self.assertAlmostEqual(by[("L-D6b", 14)]["p"], 0.5080, places=4)
        self.assertAlmostEqual(by[("L-D6a", 6)]["p"], 0.5012, places=4)
        for row in rows:
            n = row["N"]
            if n <= 6:
                self.assertAlmostEqual(row["p_ppt_bisection"], r.ghz_ppt_threshold_formula(n), places=8)
            else:
                self.assertIsNone(row["p_ppt_bisection"])
            self.assertTrue(row["one_component"])
            self.assertTrue(row["gme_certified"])
        self.assertAlmostEqual(r.ghz_ppt_threshold_formula(14), 1 / 8193, places=12)
        self.assertAlmostEqual(by[("L-D6b", 14)]["gme_sigma"], 0.89, places=2)
        self.assertAlmostEqual(by[("L-D6a", 6)]["gme_sigma"], 2.25, places=2)
        # for N = 2 the two thresholds coincide (entanglement = GME for a pair)
        self.assertAlmostEqual(r.ghz_ppt_threshold_formula(2), r.ghz_witness_threshold_formula(2), places=12)
        self.assertAlmostEqual(r.p_from_fidelity(0.5, 14), 0.49997, places=5)

    def test_claim6_threshold_gap(self):
        self.assertAlmostEqual(r.threshold_gap(2), 1.0, places=12)
        self.assertAlmostEqual(r.threshold_gap(3), 15 / 7, places=12)
        self.assertAlmostEqual(r.threshold_gap(4), 4.2, places=12)
        self.assertAlmostEqual(r.threshold_gap(14), 4096.25, places=1)
        gaps = [r.threshold_gap(n) for n in range(2, 15)]
        self.assertTrue(all(b > a for a, b in zip(gaps, gaps[1:])))
        # the witness threshold tends to 1/2 while the PPT threshold tends to 0
        self.assertLess(abs(r.ghz_witness_threshold_formula(14) - 0.5), 1e-4)
        self.assertLess(r.ghz_ppt_threshold_formula(14), 2e-4)


if __name__ == "__main__":
    unittest.main()
