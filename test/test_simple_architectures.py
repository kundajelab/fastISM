import tensorflow as tf
import unittest

from context import fastISM


class TestSimpleArchitectures(unittest.TestCase):
    def test_conv_fc(self):
        inp = tf.keras.Input((100, 4))
        x = tf.keras.layers.Conv1D(20, 3)(inp)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs=inp, outputs=x)

        fast_ism_model = fastISM.FastISM(
            model, test_correctness=False)

        self.assertTrue(fast_ism_model.test_correctness())

    def test_conv_same_padding_fc(self):
        inp = tf.keras.Input((100, 4))
        x = tf.keras.layers.Conv1D(20, 3, padding='same')(inp)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs=inp, outputs=x)

        fast_ism_model = fastISM.FastISM(
            model, test_correctness=False)

        self.assertTrue(fast_ism_model.test_correctness())

    def test_conv_even_kernel_fc(self):
        inp = tf.keras.Input((100, 4))
        x = tf.keras.layers.Conv1D(20, 4)(inp)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs=inp, outputs=x)

        fast_ism_model = fastISM.FastISM(
            model, test_correctness=False)

        self.assertTrue(fast_ism_model.test_correctness())

    def test_conv_even_kernel_same_padding_fc(self):
        inp = tf.keras.Input((100, 4))
        x = tf.keras.layers.Conv1D(20, 4, padding='same')(inp)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs=inp, outputs=x)

        fast_ism_model = fastISM.FastISM(
            model, test_correctness=False)

        self.assertTrue(fast_ism_model.test_correctness())

    def test_conv_dilated_fc(self):
        inp = tf.keras.Input((100, 4))
        x = tf.keras.layers.Conv1D(20, 3, dilation_rate=3)(inp)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs=inp, outputs=x)

        fast_ism_model = fastISM.FastISM(
            model, test_correctness=False)

        self.assertTrue(fast_ism_model.test_correctness())

    def test_conv_maxpool_fc(self):
        inp = tf.keras.Input((100, 4))
        x = tf.keras.layers.Conv1D(10, 7)(inp)
        x = tf.keras.layers.MaxPooling1D(3)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(2)(x)
        model = tf.keras.Model(inputs=inp, outputs=x)

        fast_ism_model = fastISM.FastISM(
            model, test_correctness=False)

        self.assertTrue(fast_ism_model.test_correctness())

    def test_conv_two_maxpool_fc(self):
        inp = tf.keras.Input((100, 4))
        x = tf.keras.layers.Conv1D(10, 7)(inp)
        x = tf.keras.layers.MaxPooling1D(3)(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(2)(x)
        model = tf.keras.Model(inputs=inp, outputs=x)

        fast_ism_model = fastISM.FastISM(
            model, test_correctness=False)

        self.assertTrue(fast_ism_model.test_correctness())

    def test_two_conv_maxpool_fc(self):
        inp = tf.keras.Input((100, 4))
        x = tf.keras.layers.Conv1D(10, 7, padding='same')(inp)
        x = tf.keras.layers.MaxPooling1D(3)(x)
        x = tf.keras.layers.Conv1D(10, 3)(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(2)(x)
        model = tf.keras.Model(inputs=inp, outputs=x)

        fast_ism_model = fastISM.FastISM(
            model, test_correctness=False)

        self.assertTrue(fast_ism_model.test_correctness())

    def test_four_conv_maxpool_two_fc_1(self):
        inp = tf.keras.Input((200, 4))
        x = tf.keras.layers.Conv1D(10, 7, padding='same')(inp)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.Conv1D(20, 4, padding='same')(inp)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.Conv1D(30, 2, padding='valid')(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.Conv1D(10, 6, padding='same')(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(20)(x)
        x = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs=inp, outputs=x)

        fast_ism_model = fastISM.FastISM(
            model, test_correctness=False)

        self.assertTrue(fast_ism_model.test_correctness())

    def test_four_conv_maxpool_two_fc_2(self):
        inp = tf.keras.Input((200, 4))
        x = tf.keras.layers.Conv1D(10, 3, dilation_rate=3, padding='same')(inp)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.Conv1D(
            25, 4, padding='same', activation='relu')(inp)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.Conv1D(
            30, 2, dilation_rate=2, padding='valid', activation='tanh')(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.Conv1D(10, 6, padding='same')(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(20)(x)
        x = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs=inp, outputs=x)

        fast_ism_model = fastISM.FastISM(
            model, test_correctness=False)

        self.assertTrue(fast_ism_model.test_correctness())

    def test_four_conv_maxpool_two_fc_3(self):
        inp = tf.keras.Input((200, 4))
        x = tf.keras.layers.Conv1D(10, 5, use_bias=False, padding='same')(inp)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.Conv1D(
            25, 4, padding='same', activation='relu')(inp)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.Conv1D(30, 2, dilation_rate=2, use_bias=False,
                                   padding='valid', activation='tanh')(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.Conv1D(10, 3, padding='same')(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(10)(x)
        x = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs=inp, outputs=x)

        fast_ism_model = fastISM.FastISM(
            model, test_correctness=False)

        self.assertTrue(fast_ism_model.test_correctness())


if __name__ == '__main__':
    unittest.main()
