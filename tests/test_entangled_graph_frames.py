import unittest
import numpy as np
from mapping_spaces import transformation as T
from mapping_spaces.entangled import graph_frames as G, lorentz_frames as LZ


class GraphPotentials(unittest.TestCase):
    def test_reduces_to_pass1_potentials(self):
        a = 2.0
        g = G.pass1_pair(a)
        z = np.linspace(-1.5, 1.5, 13)
        np.testing.assert_allclose(g.potential("A")(z), T.shear_potential(1, a)(z))
        np.testing.assert_allclose(g.potential("B")(z), T.shear_potential(2, a)(z))
        np.testing.assert_allclose(g.potential(G.LAB)(z), T.shear_potential(3, a)(z))

    def test_velocities_and_locality(self):
        g = G.pass1_pair(2.0)
        self.assertTrue(g.is_local("A", "B"))
        self.assertAlmostEqual(g.velocity_in_frame("A", G.LAB), -0.5)
        self.assertAlmostEqual(g.velocity_in_frame(G.LAB, "A"), 0.5)

    def test_weight_gives_linear_locality_fraction(self):
        for w in (0.0, 0.3, 0.7, 1.0):
            g = G.pass1_pair(1.0).link("A", "B", w)
            self.assertAlmostEqual(g.locality_fraction("A", "B"), w)
            self.assertAlmostEqual(g.locality_fraction("B", "A"), w)

    def test_composition_is_automatic(self):
        g = G.LocalityGraph()
        g.add(G.Particle("A", -1, -1)).add(G.Particle("B", 1, 1)).add(G.Particle("C", 0.5, 0.3))
        g.link("A", "B")
        for z in (-1, 0, 0.5, 1, 0.2):
            self.assertAlmostEqual(g.beta("A", "B", z) + g.beta("B", "C", z), g.beta("A", "C", z))

    def test_three_mutually_local_particles_in_1d(self):
        g = G.LocalityGraph()
        g.add(G.Particle("A", -1, -1)).add(G.Particle("B", 1, 1)).add(G.Particle("C", 2, 0.5))
        g.link("A", "B").link("B", "C").link("A", "C")
        for i, j in (("A", "B"), ("B", "C"), ("A", "C")):
            self.assertTrue(g.is_local(i, j))
        self.assertAlmostEqual(g.velocity_in_frame(G.LAB, "C"), -0.5)

    def test_same_height_requires_same_velocity(self):
        g = G.LocalityGraph().add(G.Particle("A", -1, -1))
        with self.assertRaises(ValueError):
            g.add(G.Particle("C", -1, 0.4))


class GraphHistory(unittest.TestCase):
    def test_swap_transfers_locality(self):
        g = G.LocalityGraph()
        # two pairs created at x=0, y=0; B and C meet again at y=2 (x=0)
        g.add(G.Particle("A", -1, -1)).add(G.Particle("B", 1, 0.0))
        g.add(G.Particle("C", -2, 0.0)).add(G.Particle("D", 2, 1))
        h = G.History(g)
        h.entangle("A", "B", 0.0).entangle("C", "D", 0.0)
        self.assertTrue(h.graph_at(1.0).is_local("A", "B"))
        self.assertFalse(h.graph_at(1.0).is_local("A", "D"))
        h.bell_measure("B", "C", 2.0)
        g2 = h.graph_at(3.0)
        self.assertTrue(g2.is_local("A", "D"))
        self.assertFalse(g2.is_local("A", "B"))
        # positions are continuous across the event in every frame
        y = np.array([1.999, 2.0, 2.001])
        for k in ("A", "D", G.LAB):
            x = h.position_in_frame("A", k, y)
            self.assertLess(abs(x[2] - x[0]), 0.01)

    def test_entangle_requires_lab_colocation(self):
        g = G.LocalityGraph().add(G.Particle("A", -1, -1)).add(G.Particle("B", 1, 1))
        with self.assertRaises(ValueError):
            G.History(g).entangle("A", "B", 1.0)


class Lorentz(unittest.TestCase):
    def test_partner_at_rest_and_proper_time(self):
        g = G.pass1_pair(a=1 / 0.6)  # v = -/+0.6
        self.assertAlmostEqual(LZ.velocity_in_frame(g, "B", "A"), 0.0)
        self.assertAlmostEqual(LZ.velocity_in_frame(g, "A", "B"), 0.0)
        (tA, xA), (tB, xB) = LZ.measurement_events_in_frame(g, "A", "A", "B", 1.0, 1.5)
        self.assertAlmostEqual(xA, 0.0)
        self.assertAlmostEqual(xB, 0.0)
        self.assertAlmostEqual(tA, LZ.proper_time(0.6, 1.0))
        self.assertAlmostEqual(tB, LZ.proper_time(0.6, 1.5))

    def test_ordering_invariant_under_global_boost(self):
        g = G.LocalityGraph().add(G.Particle("A", -1, -0.3)).add(G.Particle("B", 1, 0.8)).link("A", "B")
        first = LZ.order_in_pair_frame(g, "A", "B", 1.0, 0.7)
        for u in (-0.5, 0.2, 0.9):
            h = LZ.boosted_lab(g, u)
            # lab times of the same events in the boosted lab
            gam = 1 / np.sqrt(1 - u * u)
            tA = gam * (1.0 - u * (-0.3 * 1.0))
            tB = gam * (0.7 - u * (0.8 * 0.7))
            self.assertEqual(LZ.order_in_pair_frame(h, "A", "B", tA, tB), first)

    def test_composition(self):
        g = G.LocalityGraph().add(G.Particle("A", -1, -0.3)).add(G.Particle("B", 1, 0.8)).add(G.Particle("C", 0.4, 0.1))
        g.link("A", "B")
        t, x = 0.7, 0.2
        via = LZ.frame_to_frame(g, "B", "C", 0.4, *LZ.frame_to_frame(g, "A", "B", 0.4, t, x))
        direct = LZ.frame_to_frame(g, "A", "C", 0.4, t, x)
        np.testing.assert_allclose(via, direct)

    def test_nonrelativistic_limit_is_the_shear(self):
        eps = 1e-3
        g = G.pass1_pair(a=1 / eps)  # v = -/+eps
        z = np.linspace(-1, 1, 9)
        sigma_A = g.potential(G.LAB)(z) - np.tanh(LZ.rapidity_potential(g, "A")(z))
        np.testing.assert_allclose(sigma_A, T.shear_potential(1, 1 / eps)(z), rtol=1e-5, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
