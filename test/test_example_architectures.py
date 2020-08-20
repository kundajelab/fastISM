import unittest
import tensorflow as tf

from context import fastISM
from fastISM.models.basset import basset_model
from fastISM.models.factorized_basset import factorized_basset_model
from fastISM.models.bpnet import bpnet_model


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

    def test_bpnet_9_dilated_1000(self):
        model = bpnet_model(seqlen=1000, num_dilated_convs=9)
        
        fast_ism_model = fastISM.FastISM(
            model, test_correctness=False)

        self.assertTrue(fast_ism_model.test_correctness())


if __name__ == '__main__':
    unittest.main()
