import tensorflow as tf
import unittest

from context import fastISM


class TestCropping(unittest.TestCase):
    # many tests modified from test_simple_skip_conn_arcitectures.py
    def test_conv_crop_even_fc(self):
        # inp -> C -> Crop -> D -> y
        inp = tf.keras.Input((100, 4))
        x = tf.keras.layers.Conv1D(20, 3)(inp)
        x = tf.keras.layers.Cropping1D((4,4))(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs=inp, outputs=x)

        fast_ism_model = fastISM.FastISM(
            model, test_correctness=False)

        self.assertTrue(fast_ism_model.test_correctness())

    def test_conv_crop_odd_fc(self):
        # inp -> C -> Crop -> D -> y
        inp = tf.keras.Input((100, 4))
        x = tf.keras.layers.Conv1D(20, 3)(inp)
        x = tf.keras.layers.Cropping1D((5,3))(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs=inp, outputs=x)

        fast_ism_model = fastISM.FastISM(
            model, test_correctness=False)

        self.assertTrue(fast_ism_model.test_correctness())

    def test_conv_crop_even_odd_fc(self):
        # inp -> C -> Crop -> D -> y
        inp = tf.keras.Input((100, 4))
        x = tf.keras.layers.Conv1D(20, 3)(inp)
        x = tf.keras.layers.Cropping1D((4,5))(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs=inp, outputs=x)

        fast_ism_model = fastISM.FastISM(
            model, test_correctness=False)

        self.assertTrue(fast_ism_model.test_correctness())

    def test_conv_double_crop_fc(self):
        # inp -> C -> Crop -> Crop -> D -> y
        inp = tf.keras.Input((100, 4))
        x = tf.keras.layers.Conv1D(20, 3)(inp)
        x = tf.keras.layers.Cropping1D((2,0))(x)
        x = tf.keras.layers.Cropping1D((1,3))(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs=inp, outputs=x)

        fast_ism_model = fastISM.FastISM(
            model, test_correctness=False)

        self.assertTrue(fast_ism_model.test_correctness())

    def test_conv_crop_add_two_fc(self):
        # inp -> C -> C-> Add -> D -> y
        #          |_Crop__^
        inp = tf.keras.Input((100, 4))
        x = tf.keras.layers.Conv1D(20, 3)(inp)
        x1 = tf.keras.layers.Conv1D(20, 3, padding='valid')(x)
        x = tf.keras.layers.Cropping1D((1,1))(x)
        x = tf.keras.layers.Add()([x, x1])
        x = tf.keras.layers.Flatten()(x)
        y = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs=inp, outputs=y)

        fast_ism_model = fastISM.FastISM(
            model, test_correctness=False)

        self.assertTrue(fast_ism_model.test_correctness())

    def test_conv_add_three_fc(self):
        #          ^-C-Crop-|
        # inp -> C ->  C-> Add -> D -> y
        #          |_Crop__^
        inp = tf.keras.Input((100, 4))
        x = tf.keras.layers.Conv1D(20, 3)(inp)
        x1 = tf.keras.layers.Conv1D(20, 3, padding='valid')(x)
        x2 = tf.keras.layers.Conv1D(20, 5, padding='valid')(x)
        
        x = tf.keras.layers.Cropping1D((1,3))(x)
        x1 = tf.keras.layers.Cropping1D((2,0))(x1)
        
        x = tf.keras.layers.Add()([x, x1, x2])
        x = tf.keras.layers.Flatten()(x)
        y = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs=inp, outputs=y)

        fast_ism_model = fastISM.FastISM(
            model, test_correctness=False)

        self.assertTrue(fast_ism_model.test_correctness())

    def test_skip_crop_then_crop_mxp(self):
        #          __Crop___
        #          ^       |
        # inp -> C ->  C-> Add -> Crop -> MXP -> [without Flatten!] D -> y
        # y has output dim [_,5] per example
        inp = tf.keras.Input((100, 4))
        x = tf.keras.layers.Conv1D(20, 3)(inp)
        x1 = tf.keras.layers.Conv1D(20, 3, padding='valid')(x)
        x = tf.keras.layers.Cropping1D((0,2))(x)
        x1 = tf.keras.layers.Add()([x, x1])
        x1 = tf.keras.layers.Cropping1D((1,1))(x1)
        x2 = tf.keras.layers.MaxPooling1D(3)(x1)

        y = tf.keras.layers.Dense(5)(x2)
        model = tf.keras.Model(inputs=inp, outputs=y)

        fast_ism_model = fastISM.FastISM(
            model, test_correctness=False)

        self.assertTrue(fast_ism_model.test_correctness())

    def test_mini_dense_net_1(self):
        #          __Crop___  ___Crop____
        #          ^       |  ^         |
        # inp -> C ->  C-> Add -> C -> Add -> D -> y
        #          |________Crop________^
        inp = tf.keras.Input((100, 4))
        x = tf.keras.layers.Conv1D(20, 3)(inp)
        x1 = tf.keras.layers.Conv1D(20, 3, padding='valid')(x)
        x_crop1 = tf.keras.layers.Cropping1D((1,1))(x)
        x1 = tf.keras.layers.Add()([x_crop1, x1])
        x2 = tf.keras.layers.Conv1D(20, 5, padding='valid')(x1)

        x1 = tf.keras.layers.Cropping1D((2,2))(x1)
        x_crop2 = tf.keras.layers.Cropping1D((2,4))(x)

        x2 = tf.keras.layers.Add()([x_crop2, x1, x2])
        x2 = tf.keras.layers.Flatten()(x2)
        y = tf.keras.layers.Dense(1)(x2)
        model = tf.keras.Model(inputs=inp, outputs=y)

        fast_ism_model = fastISM.FastISM(
            model, test_correctness=False)

        self.assertTrue(fast_ism_model.test_correctness())

    def test_mini_dense_net_2(self):
        #          __Crop___  ___Crop____            _________  ___________
        #          ^       |  ^         |            ^       |  ^         |
        # inp -> C ->  C-> Add -> C -> Add -> MXP -> C -> C-> Add -> C -> Add -> D -> y        
        #          |______Crop__________^            |____________________^
        #          |_________________________Crop_________________________^
        inp = tf.keras.Input((100, 4))
        x = tf.keras.layers.Conv1D(10, 2)(inp)
        x1 = tf.keras.layers.Conv1D(10, 3, padding='valid')(x)
        x_crop1 = tf.keras.layers.Cropping1D((1,1))(x)
        x1 = tf.keras.layers.Add()([x_crop1, x1])
        x2 = tf.keras.layers.Conv1D(10, 5, padding='valid')(x1)

        x1 = tf.keras.layers.Cropping1D((2,2))(x1)
        x_crop2 = tf.keras.layers.Cropping1D((2,4))(x)

        x2 = tf.keras.layers.Add()([x_crop2, x1, x2])

        x2 = tf.keras.layers.MaxPooling1D(3)(x2)
        x2 = tf.keras.layers.Conv1D(10, 2)(x2)

        x3 = tf.keras.layers.Conv1D(10, 7, padding='same')(x2)
        x3 = tf.keras.layers.Maximum()([x2, x3])
        x4 = tf.keras.layers.Conv1D(10, 4, padding='same')(x3)

        # 99 -> 30
        x_crop3 = tf.keras.layers.Cropping1D((20,49))(x)
        x4 = tf.keras.layers.Add()([x_crop3, x2, x3, x4])

        x4 = tf.keras.layers.Flatten()(x4)
        y = tf.keras.layers.Dense(1)(x4)
        model = tf.keras.Model(inputs=inp, outputs=y)

        fast_ism_model = fastISM.FastISM(
            model, test_correctness=False)

        self.assertTrue(fast_ism_model.test_correctness())


if __name__ == '__main__':
    unittest.main()
