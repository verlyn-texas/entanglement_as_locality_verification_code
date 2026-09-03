import unittest
import numpy as np
from mapping_spaces.entangled import graph_frames as G, r11_locality_graph as r


class R11(unittest.TestCase):
    def test_claim1_reduction_to_pass1(self):
        for got, want in r.claim1_reduction().values():
            np.testing.assert_allclose(got, want, atol=1e-12)

    def test_claim2_pinned_velocities_exact(self):
        g = r.example_graph()
        V = r.claim2_velocity_table(g)
        for i in g.particles:
            for j, v in V[i].items():
                if j == i:
                    self.assertAlmostEqual(v, 0.0)
                else:
                    self.assertAlmostEqual(v, r.expected_velocity(g, i, j), places=10)
        self.assertTrue(g.is_local("A", "B") and g.is_local("C", "D"))
        self.assertFalse(g.is_local("A", "C") or g.is_local("A", "E"))

    def test_claim3_mutual_locality_in_1d(self):
        for n in (3, 4, 5):
            self.assertTrue(r.claim3_all_local(n))
            self.assertEqual(r.interpolant_degree(r.mutually_local_group(n)), n)

    def test_claim4_same_height_same_velocity(self):
        g = G.LocalityGraph().add(G.Particle("A", 1.0, 0.2))
        with self.assertRaises(ValueError):
            g.add(G.Particle("B", 1.0, 0.3))
        g.add(G.Particle("B", 1.0, 0.2))  # co-moving: allowed, and at rest mutually
        self.assertTrue(g.is_local("A", "B"))

    def test_claim5_linear_locality_fraction(self):
        ws = [0, 0.25, 0.5, 0.9, 1]
        np.testing.assert_allclose(r.claim5_locality_fraction(ws), ws, atol=1e-12)

    def test_claim6_mermin(self):
        self.assertAlmostEqual(r.claim6_mermin(), 4.0, places=12)
        self.assertEqual(r.mermin_lhv_max(), 2)


if __name__ == "__main__":
    unittest.main()
