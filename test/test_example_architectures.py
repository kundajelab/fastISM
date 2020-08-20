import tensorflow as tf
import unittest

from context import fastISM
from fastISM.models.basset import basset_model
from fastISM.models.factorized_basset import factorized_basset_model


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
        # body
        inp = tf.keras.layers.Input(shape=(500, 4))
        x = tf.keras.layers.Conv1D(
            64, kernel_size=25, padding='same', activation='relu')(inp)

        for i in range(1, 6):
            conv_x = tf.keras.layers.Conv1D(
                64, kernel_size=3, padding='same', activation='relu', dilation_rate=2**i)(x)
            x = tf.keras.layers.Add()([conv_x, x])

        bottleneck = x

        # single task
        # heads
        # profile shape head
        px = tf.keras.layers.Reshape((-1, 1, 64))(bottleneck)
        px = tf.keras.layers.Conv2DTranspose(
            2, kernel_size=(25, 1), padding='same')(px)
        # total counts head
        cx = tf.keras.layers.GlobalAvgPool1D()(bottleneck)

        model = tf.keras.Model(inputs=inp, outputs=[px, cx])

        fast_ism_model = fastISM.FastISM(
            model, test_correctness=False)

        self.assertTrue(fast_ism_model.test_correctness())

    def test_bpnet_9_dilated_1000(self):
        # body
        inp = tf.keras.layers.Input(shape=(1000, 4))
        x = tf.keras.layers.Conv1D(
            64, kernel_size=25, padding='same', activation='relu')(inp)

        for i in range(1, 10):
            conv_x = tf.keras.layers.Conv1D(
                64, kernel_size=3, padding='same', activation='relu', dilation_rate=2**i)(x)
            x = tf.keras.layers.Add()([conv_x, x])

        bottleneck = x

        # single task
        # heads
        # profile shape head
        px = tf.keras.layers.Reshape((-1, 1, 64))(bottleneck)
        px = tf.keras.layers.Conv2DTranspose(
            2, kernel_size=(25, 1), padding='same')(px)
        # total counts head
        cx = tf.keras.layers.GlobalAvgPool1D()(bottleneck)

        model = tf.keras.Model(inputs=inp, outputs=[px, cx])

        fast_ism_model = fastISM.FastISM(
            model, test_correctness=False)

        self.assertTrue(fast_ism_model.test_correctness())


if __name__ == '__main__':
    unittest.main()
