import tensorflow as tf
import unittest

from context import fastISM


class TestSimpleSingleInSingleOutArchitectures(unittest.TestCase):
    def test_conv_fc(self):
        # inp -> C -> D -> y
        inp = tf.keras.Input((100, 4))
        x = tf.keras.layers.Conv1D(20, 3)(inp)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs=inp, outputs=x)

        fast_ism_model = fastISM.FastISM(
            model, test_correctness=False)

        self.assertTrue(fast_ism_model.test_correctness())

    def test_conv_fc_sequential(self):
        # inp -> C -> D -> y
        # same as above but with Sequential
        model = tf.keras.Sequential()
        model.add(tf.keras.Input((100, 4)))
        model.add(tf.keras.layers.Conv1D(20, 3))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(1))        
                
        fast_ism_model = fastISM.FastISM(
            model, test_correctness=False)

        self.assertTrue(fast_ism_model.test_correctness())

    def test_conv_same_padding_fc(self):
        # inp -> C -> D -> y
        inp = tf.keras.Input((100, 4))
        x = tf.keras.layers.Conv1D(20, 3, padding='same')(inp)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs=inp, outputs=x)

        fast_ism_model = fastISM.FastISM(
            model, test_correctness=False)

        self.assertTrue(fast_ism_model.test_correctness())

    def test_conv_even_kernel_fc(self):
        # inp -> C -> D -> y
        inp = tf.keras.Input((100, 4))
        x = tf.keras.layers.Conv1D(20, 4)(inp)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs=inp, outputs=x)

        fast_ism_model = fastISM.FastISM(
            model, test_correctness=False)

        self.assertTrue(fast_ism_model.test_correctness())

    def test_conv_even_kernel_same_padding_fc(self):
        # inp -> C -> D -> y
        inp = tf.keras.Input((100, 4))
        x = tf.keras.layers.Conv1D(20, 4, padding='same')(inp)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs=inp, outputs=x)

        fast_ism_model = fastISM.FastISM(
            model, test_correctness=False)

        self.assertTrue(fast_ism_model.test_correctness())

    def test_conv_dilated_fc(self):
        # inp -> C -> D -> y
        inp = tf.keras.Input((100, 4))
        x = tf.keras.layers.Conv1D(20, 3, dilation_rate=3)(inp)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs=inp, outputs=x)

        fast_ism_model = fastISM.FastISM(
            model, test_correctness=False)

        self.assertTrue(fast_ism_model.test_correctness())

    def test_conv_maxpool_fc(self):
        # inp -> C -> MXP -> D -> y
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
        # inp -> C -> MXP -> MXP -> D -> y
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
        # inp -> C -> MXP -> C -> MXP -> D -> y
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
        # inp -> C -> MXP -> C -> MXP -> C -> MXP -> C -> MXP -> D -> D -> y
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
        # inp -> C -> MXP -> C -> MXP -> C -> MXP -> C -> MXP -> D -> D -> y
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
        # inp -> C -> MXP -> C -> MXP -> C -> MXP -> C -> MXP -> D -> D -> y
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
    
    def test_four_conv_maxpool_two_fc_4(self):
        # inp -> C -> MXP -> C -> MXP -> C -> MXP -> C -> MXP -> D -> D -> y
        # with Dropout and GlobalAveragePoolng1D
        inp = tf.keras.Input((200, 4))
        x = tf.keras.layers.Conv1D(10, 5, use_bias=False, padding='same')(inp)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.Conv1D(
            25, 4, padding='same', activation='relu')(inp)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.Conv1D(30, 2, dilation_rate=2, use_bias=False,
                                   padding='valid', activation='tanh')(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.Dropout(0.8)(x)
        x = tf.keras.layers.Conv1D(10, 3, padding='same')(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(10)(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs=inp, outputs=x)

        fast_ism_model = fastISM.FastISM(
            model, test_correctness=False)

        self.assertTrue(fast_ism_model.test_correctness())

    def test_four_conv_maxpool_two_fc_4_sequential(self):
        # inp -> C -> MXP -> C -> MXP -> C -> MXP -> C -> MXP -> D -> D -> y
        # with Dropout and GlobalAveragePoolng1D
        # same as above but with Sequential
        model = tf.keras.Sequential()
        model.add(tf.keras.Input((200, 4)))
        model.add(tf.keras.layers.Conv1D(10, 5, use_bias=False, padding='same'))
        model.add(tf.keras.layers.MaxPooling1D(2))
        model.add(tf.keras.layers.Conv1D(
            25, 4, padding='same', activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.MaxPooling1D(2))
        model.add(tf.keras.layers.Conv1D(30, 2, dilation_rate=2, use_bias=False,
                                   padding='valid', activation='tanh'))
        model.add(tf.keras.layers.MaxPooling1D(2))
        model.add(tf.keras.layers.Dropout(0.8))
        model.add(tf.keras.layers.Conv1D(10, 3, padding='same'))
        model.add(tf.keras.layers.MaxPooling1D(2))
        model.add(tf.keras.layers.GlobalAveragePooling1D())
        model.add(tf.keras.layers.Dense(10))
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Dense(1))

        fast_ism_model = fastISM.FastISM(
            model, test_correctness=False)

        self.assertTrue(fast_ism_model.test_correctness())


if __name__ == '__main__':
    unittest.main()
