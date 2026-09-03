import itertools
import unittest

import numpy as np

from mapping_spaces.entangled import graph_frames as G, qm
from mapping_spaces.entangled import r11_locality_graph as r11
from mapping_spaces.entangled import r13_ghz as r


class R13(unittest.TestCase):
    def test_claim1_triple_mutually_local_in_1d(self):
        g = r.triple()
        self.assertEqual(r11.interpolant_degree(g), 3)
        V = r.velocity_table(g)
        for i, j in itertools.combinations(r.NAMES, 2):
            self.assertTrue(g.is_local(i, j))
        # every particle frame sees all three particles at rest and the lab at -v_i
        for k, v in zip(r.NAMES, r.VELOCITIES):
            for j in r.NAMES:
                self.assertAlmostEqual(V[k][j], 0.0, places=12)
            self.assertAlmostEqual(V[k][G.LAB], -v, places=12)
        # the lab sees three different velocities
        self.assertEqual(len({round(V[G.LAB][j], 9) for j in r.NAMES}), 3)
        # positions: co-located (x = 0) in every particle frame for all y; fanning out in the lab
        y = np.linspace(0.0, 3.0, 7)
        pos = r.positions_in_frames(g, y)
        for k in r.NAMES:
            for j in r.NAMES:
                np.testing.assert_allclose(pos[k][j], 0.0, atol=1e-12)
        for j, v in zip(r.NAMES, r.VELOCITIES):
            np.testing.assert_allclose(pos[G.LAB][j], v * y, atol=1e-12)

    def test_claim2_ghz_signs_and_component_rule(self):
        self.assertAlmostEqual(r11.claim6_mermin(), 4.0, places=12)
        self.assertEqual(r11.mermin_lhv_max(), 2)
        s = r.ghz_signs()
        self.assertAlmostEqual(s["XXX"], 1.0, places=12)
        for k in ("XYY", "YXY", "YYX"):
            self.assertAlmostEqual(s[k], -1.0, places=12)
        n_all, best = r.lhv_assignments_satisfying()
        self.assertEqual((n_all, best), (0, 3))
        # pairwise reduced states are separable
        for c in r.pairwise_concurrences(r11.ghz()).values():
            self.assertAlmostEqual(c, 0.0, places=12)
        # one component -> three edges; the pairwise-concurrence weight gives none
        g = r.triple()
        self.assertEqual(r.entangled_components(r11.ghz(), 3), [(0, 1, 2)])
        self.assertEqual(r.edges(r.component_graph(r11.ghz(), g)), [("P1", "P2"), ("P1", "P3"), ("P2", "P3")])
        self.assertEqual(r.edges(r.pairwise_weight_graph(r11.ghz(), g)), [])
        # sanity: the component rule reproduces the pair case (singlet (x) |0>)
        psi = np.kron(qm.singlet(), qm.ket(1, 0))
        self.assertEqual(r.entangled_components(psi, 3), [(0, 1), (2,)])

    def test_claim3_measurement_decides_the_graph(self):
        g = r.triple()
        rho, p, c23, comps = r.measure_particle_1(qm.direction(0.0))  # along Z
        self.assertAlmostEqual(p, 0.5)
        self.assertAlmostEqual(c23, 0.0, places=12)
        self.assertEqual(comps, [(0,), (1,), (2,)])
        self.assertEqual(r.edges(r.component_graph(rho, g)), [])
        # postulate (ii) alone would keep the 2-3 edge
        self.assertEqual(r.edges(r.postulate_ii_graph(g, "P1")), [("P2", "P3")])
        rho, p, c23, comps = r.measure_particle_1(qm.direction(np.pi / 2))  # along X
        self.assertAlmostEqual(p, 0.5)
        self.assertAlmostEqual(c23, 1.0, places=12)
        self.assertEqual(comps, [(0,), (1, 2)])
        self.assertEqual(r.edges(r.component_graph(rho, g)), [("P2", "P3")])
        # the other outcome gives the same graphs
        for nd, want in ((qm.direction(0.0), []), (qm.direction(np.pi / 2), [("P2", "P3")])):
            self.assertEqual(r.edges(r.component_graph(r.measure_particle_1(nd, -1)[0], g)), want)
        # general direction: C_23 = |sin theta|
        thetas = np.linspace(0.0, np.pi, 9)
        np.testing.assert_allclose(r.pair_concurrence_vs_angle(thetas), np.abs(np.sin(thetas)), atol=1e-9)

    def test_claim4_continuity_and_separations(self):
        y_m = 1.0
        y = np.array([0.5, y_m, 2.0])
        v1, v2, v3 = r.VELOCITIES
        kz = r.measurement_kinematics(qm.direction(0.0), y_m, y)
        kx = r.measurement_kinematics(qm.direction(np.pi / 2), y_m, y)
        for k in (kz, kx):
            self.assertEqual(k["graph_before"], [("P1", "P2"), ("P1", "P3"), ("P2", "P3")])
            self.assertLess(k["jump"], 1e-5)  # positions continuous across the event
            # lab tracks unaffected: separation (v_j - v_i) y before and after
            np.testing.assert_allclose(k["separations"][G.LAB][("P2", "P3")], (v3 - v2) * y, atol=1e-12)
            np.testing.assert_allclose(k["separations"][G.LAB][("P1", "P2")], (v2 - v1) * y, atol=1e-12)
            # particle 1 leaves its former partners in both cases
            np.testing.assert_allclose(k["separations"]["P1"][("P1", "P2")], (v2 - v1) * np.maximum(y - y_m, 0), atol=1e-12)
        self.assertEqual(kz["graph_after"], [])
        self.assertEqual(kx["graph_after"], [("P2", "P3")])
        # Z case: 2 and 3 separate in frame 2 from y_m on, although neither was measured
        np.testing.assert_allclose(kz["separations"]["P2"][("P2", "P3")], (v3 - v2) * np.maximum(y - y_m, 0), atol=1e-12)
        self.assertAlmostEqual(kz["separations"]["P2"][("P2", "P3")][2], -0.5)
        # X case: 2 and 3 stay co-located
        np.testing.assert_allclose(kx["separations"]["P2"][("P2", "P3")], 0.0, atol=1e-12)
        # at the event the lab sees 2 and 3 apart (0.5) and 1 two units from 2
        self.assertAlmostEqual(abs(kz["separations"][G.LAB][("P2", "P3")][1]), 0.5)
        self.assertAlmostEqual(abs(kz["separations"][G.LAB][("P1", "P2")][1]), 2.0)

    def test_claim5_visibility_scaling(self):
        for V in (0.0, 0.5, 0.71, 1.0):
            self.assertAlmostEqual(r.mermin_with_visibility(V), 4 * V, places=12)
        self.assertAlmostEqual(r.mermin_with_visibility(0.5), 2.0, places=12)
        self.assertAlmostEqual(r.mermin_with_visibility(0.71), 2.84, places=12)
        self.assertAlmostEqual(r.visibility_for_mermin(2.77), 0.6925, places=12)
        self.assertAlmostEqual(r.correlator_from_fraction(0.85), 0.70, places=12)
        self.assertAlmostEqual(4 * r.correlator_from_fraction(0.85), 2.80, places=12)

    def test_claim6_n_particles_on_1d_qd(self):
        for n in (3, 6, 14):
            out = r.n_particle_check(n)
            self.assertTrue(out["all_local"])
            self.assertTrue(out["distinct_lab_velocities"])
            self.assertEqual(out["degree"], n)


if __name__ == "__main__":
    unittest.main()
