"""Tests for row R16 (relativistic, 1+1 D, c = 1): every numbered claim of
solutions/entangled_particles/rows/R16_covariant_swapping.md."""
import unittest

import numpy as np

from mapping_spaces.entangled import lorentz_frames as LZ, qm, r16_covariant_swapping as r

BOOSTS = (-0.5, 0.2, 0.9)


class R16(unittest.TestCase):
    def setUp(self):
        self.wl = r.swap_worldlines()
        self.T = r.transfer_event()

    def test_geometry(self):
        self.assertAlmostEqual(self.T[0], 2.0, places=12)
        self.assertAlmostEqual(self.T[1], 0.8, places=12)
        self.assertAlmostEqual(float(self.wl["B"].x(2.0)), float(self.wl["C"].x(2.0)), places=12)
        # a source offset keeps D's worldline and the transfer event
        wl5 = r.swap_worldlines(0.5)
        for t in (0.0, 2.0, 7.0):
            self.assertAlmostEqual(float(wl5["D"].x(t)), float(self.wl["D"].x(t)), places=12)
        self.assertAlmostEqual(float(wl5["C"].x(2.0)), 0.8, places=12)
        self.assertAlmostEqual(wl5["C"].v, -0.7, places=12)
        with self.assertRaises(ValueError):
            r.swap_worldlines(0.9)

    # ------------------------------------------------------------ claim 1
    def test_claim1_no_common_clock(self):
        wl = self.wl
        # (a) M11 frame times depend on the anchor (creation event used as origin)
        fa, fd = r.m11_frame_times(wl, "A"), r.m11_frame_times(wl, "D")
        tauA, tauD = wl["A"].proper_time(6.0), wl["D"].proper_time(7.0)
        gA, gD = 1 / np.sqrt(1 - 0.04), 1 / np.sqrt(1 - 0.25)
        self.assertAlmostEqual(fa["A"], tauA, places=12)
        self.assertAlmostEqual(fa["D"], tauD - gD * 0.5 * r.X0, places=12)      # 5.1384
        self.assertAlmostEqual(fd["D"], tauD, places=12)
        self.assertAlmostEqual(fd["A"], tauA + gA * (-0.2) * r.X0, places=12)   # 5.5522
        self.assertEqual(r.first_by(fa), "D")
        self.assertEqual(r.first_by(fd), "A")
        # (b) proper times since creation are invariant per event ...
        cc = r.creation_clocks(wl)
        self.assertAlmostEqual(cc["A"], 5.8787754, places=6)
        self.assertAlmostEqual(cc["D"], 6.0621778, places=6)
        for u in BOOSTS:
            cu = r.creation_clocks_under_boost(wl, u)
            self.assertAlmostEqual(cu["A"], cc["A"], places=12)
            self.assertAlmostEqual(cu["D"], cc["D"], places=12)
            for n in ("A", "D"):   # interval form of the same number
                self.assertAlmostEqual(
                    r.interval_proper_time((wl[n].tc, wl[n].xc), wl[n].event(r.T_MEAS[n])), cc[n], places=12)
        # ... but the order depends on when the second source fired
        self.assertEqual(r.creation_order_vs_offset((-0.5, 0.0, 0.5)), {-0.5: "A", 0.0: "A", 0.5: "D"})

    # ------------------------------------------------------------ claim 2
    def test_claim2_light_cone_clock_invariant(self):
        wl = self.wl
        cr = r.crossing_times(wl)
        self.assertAlmostEqual(cr["A"], 3.5, places=12)
        self.assertAlmostEqual(cr["D"], 5.6, places=12)
        for n in ("A", "D"):   # the crossing is lightlike from T
            ev = wl[n].event(cr[n])
            self.assertAlmostEqual(ev[0] - self.T[0], abs(ev[1] - self.T[1]), places=12)
        k = r.pair_clocks(wl)
        self.assertAlmostEqual(k["A"], 2.4494897, places=6)
        self.assertAlmostEqual(k["D"], 1.2124356, places=6)
        self.assertEqual(r.first_by(k), "D")
        self.assertEqual(r.first_by(r.creation_clocks(wl)), "A")      # differs from the creation rule
        for u in BOOSTS:
            ku = r.pair_clocks_under_boost(wl, u)
            self.assertAlmostEqual(ku["A"], k["A"], places=12)
            self.assertAlmostEqual(ku["D"], k["D"], places=12)
            self.assertEqual(r.first_by(ku), "D")
        # independent of the source offset (the creation event never enters)
        for d in (-0.5, 0.5):
            kd = r.pair_clocks(r.swap_worldlines(d))
            self.assertAlmostEqual(kd["A"], k["A"], places=12)
            self.assertAlmostEqual(kd["D"], k["D"], places=12)

    # ------------------------------------------------------------ claim 3
    def test_claim3_reduces_to_r14(self):
        c = r.clocks_when_transfer_is_creation()
        self.assertAlmostEqual(c["crossing"]["A"], 0.0, places=12)
        self.assertAlmostEqual(c["crossing"]["B"], 0.0, places=12)
        self.assertAlmostEqual(c["clock"]["A"], 0.9539392, places=6)
        self.assertAlmostEqual(c["clock"]["B"], 0.72, places=12)
        for n in ("A", "B"):
            self.assertAlmostEqual(c["clock"][n], c["r14_frame_times"][n], places=12)
        self.assertEqual(r.first_by(c["clock"]), "B")
        self.assertEqual(c["r14_first"], "B")

    # ------------------------------------------------------------ claim 4
    def test_claim4_negative_clocks_and_order_independent_statistics(self):
        wl = self.wl
        k = r.pair_clocks(wl, r.T_MEAS_EARLY)
        self.assertAlmostEqual(k["A"], -0.4898979, places=6)
        self.assertAlmostEqual(k["D"], -1.3856406, places=6)
        self.assertEqual(r.first_by(k), "D")
        off = r.light_travel_offsets(wl)
        self.assertAlmostEqual(off["A"], 1.5, places=12)
        self.assertAlmostEqual(off["D"], 3.6, places=12)
        # sign of the clock = "inside the future cone of T", in every frame
        rng = np.random.default_rng(16)
        for _ in range(200):
            w = r.Worldline("W", 1.0, rng.uniform(-0.9, 0.9), rng.uniform(-1, 1), rng.uniform(-2, 2))
            T = (rng.uniform(-1, 1), rng.uniform(-2, 2))
            t = rng.uniform(-3, 5)
            kap = r.pair_clock(w, T, t)
            self.assertEqual(kap >= 0, r.in_future_cone(w.event(t), T))
            u = rng.uniform(-0.9, 0.9)
            self.assertAlmostEqual(r.pair_clock(w.boosted(u), r.boost_event(u, T), r.boost_event(u, w.event(t))[0]),
                                   kap, places=9)
        # statistics: either order gives the QM box, S = 2 sqrt 2, no signalling
        for order in ("A", "D"):
            _, S, sig, dev = r.swapped_pair_box(order)
            self.assertAlmostEqual(S, qm.TSIRELSON, places=12)
            self.assertLess(sig, 1e-12)
            self.assertLess(dev, 1e-12)

    # ------------------------------------------------------------ claim 5
    def test_claim5_lab_simultaneity_is_frame_dependent(self):
        wl = self.wl
        k0 = r.pair_clocks(wl, rule=r.lab_sync_clock)
        self.assertAlmostEqual(k0["A"], 3.9191836, places=6)
        self.assertAlmostEqual(k0["D"], 4.3301270, places=6)
        firsts = {}
        for u in (-0.5, 0.0, 0.5):
            ku = r.pair_clocks_under_boost(wl, u, rule=r.lab_sync_clock)
            firsts[u] = r.first_by(ku)
        self.assertEqual(firsts, {-0.5: "A", 0.0: "A", 0.5: "D"})
        km = r.pair_clocks_under_boost(wl, -0.5, rule=r.lab_sync_clock)
        kp = r.pair_clocks_under_boost(wl, 0.5, rule=r.lab_sync_clock)
        self.assertAlmostEqual(km["A"], 3.2659863, places=6)
        self.assertAlmostEqual(kp["D"], 3.2908965, places=6)

    # ------------------------------------------------------------ claim 6
    def test_claim6_ghz_triple_invariant_order(self):
        g = r.ghz_triple()
        for k in r.GHZ_NAMES:
            ev = r.ghz_frame_events(g, k)
            for n, t, v in zip(r.GHZ_NAMES, r.GHZ_T_MEAS, r.GHZ_VELOCITIES):
                self.assertAlmostEqual(ev[n][1], 0.0, places=12)
                self.assertAlmostEqual(ev[n][0], LZ.proper_time(v, t), places=12)
        order0, times0, _ = r.ghz_order_under_boost(0.0)
        self.assertEqual(order0, ("P2", "P1", "P3"))
        np.testing.assert_allclose([times0[n] for n in r.GHZ_NAMES], [0.8, 0.72, 1.0493331], atol=1e-6)
        for u in BOOSTS:
            order, times, xs = r.ghz_order_under_boost(u)
            self.assertEqual(order, order0)
            for n in r.GHZ_NAMES:
                self.assertAlmostEqual(times[n], times0[n], places=12)
                self.assertAlmostEqual(xs[n], 0.0, places=12)
        lc = r.ghz_light_cone_clocks()
        for n in r.GHZ_NAMES:
            self.assertAlmostEqual(lc[n], times0[n], places=12)

    # ------------------------------------------------------------ claim 7
    def test_claim7_delayed_choice_clocks_negative_in_every_frame(self):
        for d, expect in ((1.0, -488.336), (10.0, -518.356), (100.0, -818.564)):
            out = r.ld2_clocks(d)
            for beta, row in out.items():
                for n in ("A", "D"):
                    self.assertAlmostEqual(row[n], expect, places=2)
                    self.assertFalse(row[n + "_in_future_cone"])
        self.assertAlmostEqual(r.ld2_cone_boundary_metres(), 145.399, places=2)


if __name__ == "__main__":
    unittest.main()


class TestChartJump(unittest.TestCase):
    """Round-5 H4: the covariant chart jump at re-anchoring, made explicit."""

    def test_positions_in_A_chart_at_the_cone_crossing(self):
        out = r.chart_jump_at_reanchoring()
        self.assertAlmostEqual(out["t_crossing"], 3.5, places=9)
        self.assertAlmostEqual(out["B"]["before"], 0.0, places=9)
        self.assertAlmostEqual(out["B"]["after"], 2.1433, places=3)
        self.assertAlmostEqual(out["D"]["before"], 4.1335, places=3)
        self.assertAlmostEqual(out["D"]["after"], 1.6 / np.sqrt(1 - 0.25), places=9)   # 1.8475
