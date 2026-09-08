"""R25 — the repaired covariant ordering rule (round-4 revision, B1/B2).

The first two test classes port the FAIL lines of the round-4 referee scripts
(check_p4_anchor.py, check_C_kappa_causal.py) and require them to pass under
the repaired rule; the rest verify the theorem (monotone in the cone, strict
partial order, causal consistency, invariance) on random configurations with
creation, transfer and pruning anchors, and the paper's worked numbers.
"""
import itertools
import unittest

import numpy as np

from mapping_spaces.entangled import r16_covariant_swapping as r16
from mapping_spaces.entangled import r25_ordering_rule as r25


class TestRefereeCounterexamples(unittest.TestCase):
    def test_referee_B_ghz_pruning_now_causal(self):
        out = r25.referee_B_ghz_pruning()
        self.assertEqual(out["causal"], 1)                 # 2 before 3
        self.assertEqual(out["old_order"], "3 first")      # v4's defect reproduced
        self.assertEqual(out["new_order"], "2 first")
        self.assertTrue(out["new_agrees_with_causal"])
        self.assertLess(out["new_clocks"][0], 0.0)         # E2 outside E1's cone
        self.assertGreater(out["new_clocks"][1], 0.0)

    def test_referee_C_instance_now_causal(self):
        out = r25.referee_C_instance()
        self.assertEqual(out["causal"], 1)                 # A before D
        self.assertAlmostEqual(out["old_clocks"][0], 1.900, places=3)
        self.assertAlmostEqual(out["old_clocks"][1], 0.803, places=3)
        self.assertEqual(out["old_order"], "D first")
        self.assertEqual(out["new_order"], "A first")

    def test_referee_B_worked_instance_now_causal(self):
        out = r25.referee_B_worked_instance()
        self.assertTrue(out["causal_2_before_1"])
        self.assertTrue(out["old_inverted"])
        self.assertTrue(out["new_2_before_1"])

    def test_no_timelike_inversion_under_new_rule(self):
        out = r25.random_transfer_pairs(200000)
        self.assertGreater(out["timelike_pairs"], 40000)
        self.assertGreater(out["fraction_old"], 0.02)      # the defect was generic
        self.assertEqual(out["inverted_new"], 0)


class TestTheorem(unittest.TestCase):
    def test_kappa_monotone_along_causal_curves_in_cone(self):
        out = r25.kappa_monotone_in_cone()
        self.assertTrue(out["monotone"])

    def test_no_invariant_clock_outside_cone(self):
        out = r25.no_invariant_time_function_outside_cone()
        self.assertAlmostEqual(out["kappa_1"], out["kappa_2"], places=9)
        self.assertTrue(out["causal_1_before_2"])

    def test_random_components_partial_order_causal_invariant(self):
        for members in (2, 3, 4):
            out = r25.random_components(600, members=members, seed=25 + members)
            self.assertEqual(out["not_partial_order"], 0, out)
            self.assertEqual(out["causal_violations"], 0, out)
            self.assertEqual(out["not_invariant"], 0, out)

    def test_kappa_invariant_under_boosts(self):
        rng = np.random.default_rng(1)
        for _ in range(300):
            E = (rng.uniform(-2, 2), rng.uniform(-2, 2))
            e = (rng.uniform(-4, 4), rng.uniform(-4, 4))
            k = r25.signed_interval(E, e)
            for u in (-0.7, 0.2, 0.9):
                kb = r25.signed_interval(r25.boost_event(u, E), r25.boost_event(u, e))
                self.assertAlmostEqual(k, kb, places=9)
            self.assertEqual(r25.in_future_cone(e, E), k > 0 or (abs(k) < 1e-12 and e[0] >= E[0]))

    def test_reduces_to_creation_rule(self):
        self.assertLess(r25.creation_rule_reduction(), 1e-12)

    def test_ghz_triple_three_spacelike_events(self):
        out = r25.ghz_spacelike_triple(t_meas=(1.0, 1.05, 0.9))
        self.assertTrue(out["mutually_spacelike"])
        self.assertTrue(out["total"])
        self.assertTrue(out["strict_partial_order"])
        self.assertTrue(out["invariant"])
        np.testing.assert_allclose(out["kappa"], out["proper_times"], atol=1e-12)
        # tie on the symmetric configuration: unordered, still a partial order
        tie = r25.ghz_spacelike_triple(t_meas=(1.0, 1.05, 1.0))
        self.assertFalse(tie["total"])
        self.assertTrue(tie["strict_partial_order"])


class TestWorkedNumbers(unittest.TestCase):
    def test_working_pair(self):
        out = r25.working_pair_clocks()
        late = out["late"]
        self.assertAlmostEqual(late["kappa"]["A"], 2 * np.sqrt(3), places=9)     # 3.4641
        self.assertAlmostEqual(late["kappa"]["D"], np.sqrt(6.51), places=9)      # 2.5515
        self.assertEqual(late["first"], "D")
        self.assertTrue(late["spacelike"] and late["invariant"])
        self.assertTrue(all(late["in_cone"].values()))
        # Eq. (10)'s values, for the record
        self.assertAlmostEqual(late["kappa_eq10"]["A"], 2.4494897, places=6)
        self.assertAlmostEqual(late["kappa_eq10"]["D"], 1.2124356, places=6)
        early = out["early"]
        self.assertAlmostEqual(early["kappa"]["A"], -np.sqrt(0.96), places=9)    # -0.9798
        self.assertAlmostEqual(early["kappa"]["D"], -np.sqrt(3.84), places=9)    # -1.9596
        self.assertFalse(any(early["in_cone"].values()))
        self.assertIsNone(early["first"])                                        # unordered outside the cone

    def test_delayed_choice_geometry(self):
        for d, expect in ((1.0, -484.99), (10.0, -483.85), (100.0, -352.08)):
            out = r25.delayed_choice_clocks(d)
            for beta, row in out.items():
                self.assertAlmostEqual(row["kappa"][0], expect, places=1, msg=(d, beta))
                self.assertAlmostEqual(row["kappa"][1], expect, places=1, msg=(d, beta))
                self.assertEqual(row["in_cone"], (False, False))
                self.assertFalse(row["ordered"])

    def test_nonrelativistic_window(self):
        out = r25.nonrelativistic_window(1e4, 2e4)
        self.assertAlmostEqual(out["window"] / 1.67e-9, 1.0, delta=0.01)
        self.assertLess(out["undetermined"], 1e-10)
        self.assertGreater(out["undetermined"], 1e-11)


if __name__ == "__main__":
    unittest.main()


class TestRound5Anchoring(unittest.TestCase):
    """Round-5 T2 (inherited anchor for a pruned component) and T3 (the
    component orders do not compose)."""

    def test_pruned_pair_inherits_the_parent_anchor(self):
        out = r25.pruned_pair_anchors()
        self.assertTrue(out["on_worldlines"])
        self.assertEqual(out["first"], 0)
        self.assertEqual(out["remnant"], [1, 2])
        np.testing.assert_allclose(out["kappa_O"], [1.0, 2.322, 2.156], atol=1e-3)
        self.assertEqual(out["inherited_first"], 2)            # 3 before 2 by kappa from O
        np.testing.assert_allclose(out["kappa_pruning"], [0.624, 1.116], atol=1e-3)
        self.assertEqual(out["pruning_first"], 1)              # the withdrawn reading: 2 before 3
        rnd = r25.random_spacelike_triples_pruning_anchor(1000)
        self.assertEqual(rnd["unordered_inherited"], 0)        # total order by proper time from O
        self.assertEqual(rnd["unordered_pruning"], rnd["triples"])   # all of them (round-6 J9)

    def test_component_orders_do_not_compose(self):
        out = r25.cross_component_cycle()
        self.assertTrue(out["MA_T_spacelike"] and out["T_MD_spacelike"])
        self.assertAlmostEqual(out["kappa_E1"]["MA"], 0.8529, places=4)
        self.assertAlmostEqual(out["kappa_E1"]["T"], 0.8660, places=4)
        self.assertAlmostEqual(out["kappa_E2"]["MD"], 0.8718, places=4)
        self.assertTrue(out["A_before_T_in_AB"])
        self.assertTrue(out["T_before_D_in_CD"])
        self.assertTrue(out["MD_causally_before_MA"])
        self.assertTrue(out["cyclic"])
        self.assertTrue(out["component_orders_strict"])
        self.assertTrue(all(abs(v) < 1 for v in out["velocities"].values()))
        rate = r25.cross_component_cycle_rate(40000)
        self.assertGreater(rate["cycles"], 0)
        self.assertLess(rate["rate"], 0.01)
        self.assertEqual(rate["cycles_outside_delayed_choice"], 0)


class TestRound6Precisions(unittest.TestCase):
    """Round-6 item J8 (C-P2): iterated swapping with a transfer-anchored
    parent can leave successor-component membership P4-undefined."""

    def test_iterated_swapping_membership_unordered(self):
        out = r25.iterated_swapping_unordered()
        self.assertAlmostEqual(out["kappa_MA"], -0.980, places=3)
        self.assertAlmostEqual(out["kappa_T2"], 1.249, places=3)
        self.assertFalse(out["MA_in_cone"])
        self.assertTrue(out["T2_in_cone"])
        self.assertTrue(out["spacelike"])
        self.assertIsNone(out["first"])
