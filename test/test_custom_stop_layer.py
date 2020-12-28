import tensorflow as tf
import unittest

from context import fastISM
from fastISM.models.bpnet import bpnet_model


class TestCustomStopLayer(unittest.TestCase):
    # testing introducing stop layers at intermediate nodes (early_stop_layers)
    # that are not necessarily in STOP_LAYERS (i.e. not a dense/flatten etc
    # layer) could be conv/add layers as well. Would be useful if perturbed
    # width increases quickly at spans large fraction of intermediate conv
    # layers

    def test_two_conv_addstop_fc(self):
        # inp --> C -> Add (stop) -> D -> y
        #     |-> C ----^
        inp = tf.keras.Input((100, 4))
        x1 = tf.keras.layers.Conv1D(20, 3)(inp)
        x2 = tf.keras.layers.Conv1D(20, 3)(inp)

        stop_add = tf.keras.layers.Add()
        x = stop_add([x1, x2])

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs=inp, outputs=x)

        fast_ism_model = fastISM.FastISM(
            model,
            early_stop_layers=stop_add.name,
            test_correctness=False)

        self.assertTrue(fast_ism_model.test_correctness())

    def test_two_conv_addstop_skip_fc(self):
        # inp --> C -> Add (stop) -> D --> Add -> y
        #     |-> C -> ^ ----------> D -----^
        inp = tf.keras.Input((100, 4))
        x1 = tf.keras.layers.Conv1D(20, 3)(inp)
        x2 = tf.keras.layers.Conv1D(20, 3)(inp)

        stop_add = tf.keras.layers.Add()
        x = stop_add([x1, x2])

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(10)(x)

        x2 = tf.keras.layers.Flatten()(x2)
        x2 = tf.keras.layers.Dense(10)(x2)

        x = tf.keras.layers.Add()([x, x2])

        model = tf.keras.Model(inputs=inp, outputs=x)

        fast_ism_model = fastISM.FastISM(
            model,
            early_stop_layers=stop_add.name,
            test_correctness=False)

        self.assertTrue(fast_ism_model.test_correctness())

    def test_conv_into_stop_segment(self):
        # inp --> C -> C (stop) -> Add -> D --> y
        #         |--> C -----------^
        inp = tf.keras.Input((100, 4))
        x = tf.keras.layers.Conv1D(20, 3)(inp)

        x1 = tf.keras.layers.Conv1D(20, 3)(x)

        stop_conv = tf.keras.layers.Conv1D(20, 3)
        x = stop_conv(x)

        x = tf.keras.layers.Add()([x, x1])

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1)(x)

        model = tf.keras.Model(inputs=inp, outputs=x)

        fast_ism_model = fastISM.FastISM(
            model,
            early_stop_layers=stop_conv.name,
            test_correctness=False)

        self.assertTrue(fast_ism_model.test_correctness())

    def test_two_conv_maxpool_fc(self):
        # inp -> C -> MXP -> C -> MXP -> D -> y
        inp = tf.keras.Input((100, 4))
        x = tf.keras.layers.Conv1D(10, 7, padding='same')(inp)
        x = tf.keras.layers.MaxPooling1D(3)(x)
        x = tf.keras.layers.Conv1D(10, 3)(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(2)(x)
        model = tf.keras.Model(inputs=inp, outputs=x)

        for layer in model.layers[1:5]:
            fast_ism_model = fastISM.FastISM(
                model,
                early_stop_layers=layer.name,
                test_correctness=False)

            self.assertTrue(fast_ism_model.test_correctness())

    def test_mini_dense_net(self):
        # early stops added at layers with "x"
        #          _________  _____________                    __________________    _____________
        #          ^       |  ^            |                   ^                 |   ^            |
        # inp -> C ->  C-> Add1 (x)-> C -> Add2(x) -> MXP (x) -> C1 (x) -> C2 -> Max1 (x) -> C -> Add -> D -> y
        #          |_______________________^                  |___________________________________^
        inp = tf.keras.Input((100, 4))
        x = tf.keras.layers.Conv1D(20, 3)(inp)
        x1 = tf.keras.layers.Conv1D(20, 3, padding='same')(x)

        add1 = tf.keras.layers.Add()
        x1 = add1([x, x1])

        x2 = tf.keras.layers.Conv1D(20, 5, padding='same')(x1)

        add2 = tf.keras.layers.Add()
        x2 = add2([x, x1, x2])

        mxp = tf.keras.layers.MaxPooling1D(3)
        x2 = mxp(x2)

        c1 = tf.keras.layers.Conv1D(10, 2)
        x2 = c1(x2)

        x3 = tf.keras.layers.Conv1D(10, 7, padding='same')(x2)

        max1 = tf.keras.layers.Maximum()
        x3 = max1([x2, x3])

        x4 = tf.keras.layers.Conv1D(10, 4, padding='same')(x3)
        x4 = tf.keras.layers.Add()([x2, x3, x4])

        x4 = tf.keras.layers.Flatten()(x4)
        y = tf.keras.layers.Dense(1)(x4)
        model = tf.keras.Model(inputs=inp, outputs=y)

        for layer in [add1, add2, mxp, c1, max1]:
            fast_ism_model = fastISM.FastISM(
                model,
                early_stop_layers=layer.name,
                test_correctness=False)

            self.assertTrue(fast_ism_model.test_correctness())

    def test_bpnet_5_dilated_100(self):
        model = bpnet_model(seqlen=100, num_dilated_convs=5)

        conv_layers = [x.name for x in model.layers if 'conv1d' in x.name]

        for conv_layer in conv_layers:
            # try with an early stop at each of conv layers
            fast_ism_model = fastISM.FastISM(
                model,
                early_stop_layers=conv_layer,
                test_correctness=False)

            # seems to need lower numerical to always pass
            self.assertTrue(fast_ism_model.test_correctness(atol=1e-5))

    def test_bpnet_9_dilated_100(self):
        model = bpnet_model(seqlen=100, num_dilated_convs=9)

        conv_layers = [x.name for x in model.layers if 'conv1d' in x.name]

        for conv_layer in conv_layers[-4:]:
            # try with an early stop at each of the last 4 conv layers
            fast_ism_model = fastISM.FastISM(
                model,
                early_stop_layers=conv_layer,
                test_correctness=False)

            # seems to need lower numerical to always pass
            self.assertTrue(fast_ism_model.test_correctness(atol=1e-5))


if __name__ == '__main__':
    unittest.main()
