import tensorflow as tf
import unittest

from context import fastISM


class TestSimpleSingleInMultiOutArchitectures(unittest.TestCase):
    def test_conv_two_fc(self):
        #         /- D -> y1
        # inp -> C
        #         \_ D -> y2
        inp = tf.keras.Input((100, 4))
        x = tf.keras.layers.Conv1D(20, 3)(inp)
        x = tf.keras.layers.Flatten()(x)
        y1 = tf.keras.layers.Dense(1)(x)
        y2 = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs=inp, outputs=[y1, y2])

        fast_ism_model = fastISM.FastISM(
            model, test_correctness=False)

        self.assertTrue(fast_ism_model.test_correctness())

    def test_conv_three_fc(self):
        #         /- D -> y1
        # inp -> C - D -> y2
        #         \_ D -> y3
        inp = tf.keras.Input((100, 4))
        x = tf.keras.layers.Conv1D(20, 3)(inp)
        x = tf.keras.layers.Flatten()(x)
        y1 = tf.keras.layers.Dense(1)(x)
        y2 = tf.keras.layers.Dense(1)(x)
        y3 = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs=inp, outputs=[y1, y2, y3])

        fast_ism_model = fastISM.FastISM(
            model, test_correctness=False)

        self.assertTrue(fast_ism_model.test_correctness())

    def test_conv_fc_two_head(self):
        # inp -> C -> D -> D -> y1
        #                  \_ D -> y2
        inp = tf.keras.Input((100, 4))
        x = tf.keras.layers.Conv1D(20, 3)(inp)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(10)(x)
        y1 = tf.keras.layers.Dense(1)(x)
        y2 = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs=inp, outputs=[y1, y2])

        fast_ism_model = fastISM.FastISM(
            model, test_correctness=False)

        self.assertTrue(fast_ism_model.test_correctness())

    def test_two_conv_fc_per_conv(self):
        #         /- D -> y1
        # inp -> C
        #         \_ C -> D -> y2
        inp = tf.keras.Input((100, 4))
        x1 = tf.keras.layers.Conv1D(20, 3)(inp)
        x2 = tf.keras.layers.Conv1D(20, 3)(x1)
        x1f = tf.keras.layers.Flatten()(x1)
        x2f = tf.keras.layers.Flatten()(x2)
        y1 = tf.keras.layers.Dense(1)(x1f)
        y2 = tf.keras.layers.Dense(1)(x2f)
        model = tf.keras.Model(inputs=inp, outputs=[y1, y2])

        fast_ism_model = fastISM.FastISM(
            model, test_correctness=False)

        self.assertTrue(fast_ism_model.test_correctness())

    def test_three_conv_maxpool_fc_per_conv(self):
        #              /- D -> y1
        # inp -> C -> MX -> C -> MX -> C -> MX -> D -> y2
        #                          \_ C -> D -> y3
        inp = tf.keras.Input((100, 4))
        x1 = tf.keras.layers.Conv1D(20, 3, padding='same')(inp)
        x1 = tf.keras.layers.MaxPool1D(2)(x1)
        x2 = tf.keras.layers.Conv1D(10, 4, padding='same')(x1)
        x2 = tf.keras.layers.MaxPool1D(2)(x2)
        x3 = tf.keras.layers.Conv1D(10, 3)(x2)
        x3 = tf.keras.layers.MaxPool1D(3)(x3)
        x1f = tf.keras.layers.Flatten()(x1)
        x2f = tf.keras.layers.Flatten()(x2)
        x3f = tf.keras.layers.Flatten()(x3)
        y1 = tf.keras.layers.Dense(1)(x1f)
        y2 = tf.keras.layers.Dense(1)(x2f)
        y3 = tf.keras.layers.Dense(1)(x3f)
        model = tf.keras.Model(inputs=inp, outputs=[y1, y2, y3])

        fast_ism_model = fastISM.FastISM(
            model, test_correctness=False)

        self.assertTrue(fast_ism_model.test_correctness())

    def test_input_split_conv_fc(self):
        #    /- C -> D -> y1
        # inp
        #    \_ C -> D -> y2
        inp = tf.keras.Input((100, 4))
        x1 = tf.keras.layers.Conv1D(20, 3)(inp)
        x2 = tf.keras.layers.Conv1D(10, 4)(inp)
        x1f = tf.keras.layers.Flatten()(x1)
        x2f = tf.keras.layers.Flatten()(x2)
        y1 = tf.keras.layers.Dense(1)(x1f)
        y2 = tf.keras.layers.Dense(1)(x2f)
        model = tf.keras.Model(inputs=inp, outputs=[y1, y2])

        fast_ism_model = fastISM.FastISM(
            model, test_correctness=False)

        self.assertTrue(fast_ism_model.test_correctness())

    def test_input_split_complex(self):
        #    /- C -> MXP -> C -> MXP -> D -> y1
        # inp          \_ C -> MXP -> D -> D -> y2
        #    \_ C -> MXP -> D -> y3
        inp = tf.keras.Input((100, 4))

        # first row
        x1 = tf.keras.layers.Conv1D(20, 3, dilation_rate=2)(inp)
        x1 = tf.keras.layers.MaxPooling1D(2)(x1)
        x11 = tf.keras.layers.Conv1D(20, 3, dilation_rate=3)(x1)
        x11 = tf.keras.layers.MaxPooling1D(2)(x11)
        x11f = tf.keras.layers.Flatten()(x11)
        y1 = tf.keras.layers.Dense(5)(x11f)

        # second row
        x12 = tf.keras.layers.Conv1D(
            15, 2, padding='same', activation='relu')(x1)
        x12 = tf.keras.layers.MaxPooling1D(2)(x12)
        x12f = tf.keras.layers.Flatten()(x12)
        y2 = tf.keras.layers.Dense(5)(x12f)
        y2 = tf.keras.layers.Dense(2, activation='tanh')(y2)

        # third row
        x2 = tf.keras.layers.Conv1D(10, 4, padding='same')(inp)
        x2 = tf.keras.layers.MaxPool1D(3)(x2)
        x2f = tf.keras.layers.Flatten()(x2)
        y3 = tf.keras.layers.Dense(1)(x2f)

        model = tf.keras.Model(inputs=inp, outputs=[y1, y2, y3])

        fast_ism_model = fastISM.FastISM(
            model, test_correctness=False)

        self.assertTrue(fast_ism_model.test_correctness())

    def test_input_split_complex_10bp_change_range(self):
        #    /- C -> MXP -> C -> MXP -> D -> y1
        # inp          \_ C -> MXP -> D -> D -> y2
        #    \_ C -> MXP -> D -> y3
        inp = tf.keras.Input((100, 4))

        # first row
        x1 = tf.keras.layers.Conv1D(20, 3, dilation_rate=2)(inp)
        x1 = tf.keras.layers.MaxPooling1D(2)(x1)
        x11 = tf.keras.layers.Conv1D(20, 3, dilation_rate=3)(x1)
        x11 = tf.keras.layers.MaxPooling1D(2)(x11)
        x11f = tf.keras.layers.Flatten()(x11)
        y1 = tf.keras.layers.Dense(5)(x11f)

        # second row
        x12 = tf.keras.layers.Conv1D(
            15, 2, padding='same', activation='relu')(x1)
        x12 = tf.keras.layers.MaxPooling1D(2)(x12)
        x12f = tf.keras.layers.Flatten()(x12)
        y2 = tf.keras.layers.Dense(5)(x12f)
        y2 = tf.keras.layers.Dense(2, activation='tanh')(y2)

        # third row
        x2 = tf.keras.layers.Conv1D(10, 4, padding='same')(inp)
        x2 = tf.keras.layers.MaxPool1D(3)(x2)
        x2f = tf.keras.layers.Flatten()(x2)
        y3 = tf.keras.layers.Dense(1)(x2f)

        model = tf.keras.Model(inputs=inp, outputs=[y1, y2, y3])

        fast_ism_model = fastISM.FastISM(
            model, change_ranges=[(i, i+10) for i in range(0, 100, 10)],
            test_correctness=False)

        self.assertTrue(fast_ism_model.test_correctness())


if __name__ == '__main__':
    unittest.main()
