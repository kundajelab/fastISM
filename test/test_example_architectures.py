import unittest
import tensorflow as tf

from context import fastISM
from fastISM.models.basset import basset_model
from fastISM.models.factorized_basset import factorized_basset_model
from fastISM.models.bpnet import bpnet_model
from fastISM.models.bpnet_dense import bpnet_dense_model
from random import sample

# Takes a few mins!


class TestExampleArchitectures(unittest.TestCase):
    def test_basset_200(self):
        model = basset_model(seqlen=200)

        fast_ism_model = fastISM.FastISM(
            model, test_correctness=False)

        self.assertTrue(fast_ism_model.test_correctness())

    def test_basset_500(self):
        model = basset_model(seqlen=500)

        fast_ism_model = fastISM.FastISM(
            model, test_correctness=False)

        self.assertTrue(fast_ism_model.test_correctness())

    def test_factorized_basset_200(self):
        model = factorized_basset_model(seqlen=200)

        fast_ism_model = fastISM.FastISM(
            model, test_correctness=False)

        self.assertTrue(fast_ism_model.test_correctness())

    def test_factorized_basset_500(self):
        model = factorized_basset_model(seqlen=500)

        fast_ism_model = fastISM.FastISM(
            model, test_correctness=False)

        self.assertTrue(fast_ism_model.test_correctness())

    def test_bpnet_5_dilated_500(self):
        model = bpnet_model(seqlen=500, num_dilated_convs=5)

        fast_ism_model = fastISM.FastISM(
            model, test_correctness=False)

        self.assertTrue(fast_ism_model.test_correctness())

    def test_bpnet_9_dilated_100(self):
        model = bpnet_model(seqlen=100, num_dilated_convs=9)

        fast_ism_model = fastISM.FastISM(
            model, test_correctness=False)

        # seems to need lower numerical to always pass
        self.assertTrue(fast_ism_model.test_correctness(atol=1e-5))

    def test_bpnet_9_dilated_500(self):
        model = bpnet_model(seqlen=500, num_dilated_convs=9)

        fast_ism_model = fastISM.FastISM(
            model, test_correctness=False)

        # seems to need lower numerical to always pass
        self.assertTrue(fast_ism_model.test_correctness(atol=1e-5))

    def test_bpnet_dense_2_672(self):
        model = bpnet_dense_model(inlen=672, outlen=500, ndl=2)

        # too slow for all, so select randomly (preference for edges)
        change_ranges = [(x, x+1) for x in range(0, 10)] + \
            [(x, x+1) for x in sorted(sample(range(10, 662), 50))] + \
            [(x, x+1) for x in range(662, 672)]

        fast_ism_model = fastISM.FastISM(
            model,
            change_ranges=change_ranges,
            early_stop_layers='profile_out_prebias',
            test_correctness=False)

        # seems to need lower numerical to always pass
        self.assertTrue(fast_ism_model.test_correctness(atol=1e-5))

    def test_bpnet_dense_6_1346(self):
        model = bpnet_dense_model(inlen=1346, outlen=1000, filters=8, ndl=6)

        # too slow for all, so select randomly (preference for edges)
        change_ranges = [(x, x+1) for x in range(0, 10)] + \
            [(x, x+1) for x in sorted(sample(range(10, 1336), 50))] + \
            [(x, x+1) for x in range(1336, 1346)]

        fast_ism_model = fastISM.FastISM(
            model,
            change_ranges=change_ranges,
            early_stop_layers='profile_out_prebias',
            test_correctness=False)

        # seems to need lower numerical to always pass
        self.assertTrue(fast_ism_model.test_correctness(atol=1e-5))


if __name__ == '__main__':
    unittest.main()
