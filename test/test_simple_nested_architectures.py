import tensorflow as tf
import unittest

from context import fastISM

def conv_block(input_shape=(108,4)):
    inp = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(20, 3, padding='same')(inp)
    x = tf.keras.layers.MaxPooling1D(3)(x)
    x = tf.keras.layers.Conv1D(20, 5, padding='same')(x)
    x = tf.keras.layers.MaxPooling1D(3)(x)
    x = tf.keras.layers.Conv1D(20, 9, padding='same')(x)
    x = tf.keras.layers.MaxPooling1D(3)(x)
    model = tf.keras.Model(inputs=inp, outputs=x)
    return model

def res_block(input_shape=(108,20)):
    inp = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(20, 3, padding='same')(inp)    
    x = tf.keras.layers.Add()([inp, x])
    model = tf.keras.Model(inputs=inp, outputs=x)
    return model

def doub_res_block(input_shape=(108,20)):
    inp = tf.keras.Input(shape=input_shape)
    x = res_block()(inp)
    x = res_block()(x)    
    model = tf.keras.Model(inputs=inp, outputs=x)
    return model    

def fc_block(input_shape=(80,)):
    inp = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dense(10)(inp)
    x = tf.keras.layers.Dense(1)(x)
    
    model = tf.keras.Model(inputs=inp, outputs=x)
    return model

class TestSimpleSingleNestedArchitectures(unittest.TestCase):
    def test_three_conv_two_fc(self):
        # inp -> [ C -> M -> C -> M -> C -> M ] -> [ D -> D -> y ]
        convs = conv_block()
        fcs = fc_block()

        inp = tf.keras.Input((108, 4))
        x = convs(inp)
        x = tf.keras.layers.Flatten()(x)
        x = fcs(x)
        
        model = tf.keras.Model(inputs=inp, outputs=x)

        fast_ism_model = fastISM.FastISM(
            model, test_correctness=False)

        self.assertTrue(fast_ism_model.test_correctness())
    
    def test_conv_res_mxp_two_fc(self):
        #            _________
        #            ^       |
        # inp -> C [ ->  C -> Add ] -> M -> [ D -> D -> y ]
        res = res_block()
        fcs = fc_block(input_shape=(36*20,))

        inp = tf.keras.Input((108, 4))
        x = tf.keras.layers.Conv1D(20, 3, padding='same')(inp)
        x = res(x)
        x = tf.keras.layers.MaxPooling1D(3)(x)
        x = tf.keras.layers.Flatten()(x)
        x = fcs(x)
        
        model = tf.keras.Model(inputs=inp, outputs=x)

        fast_ism_model = fastISM.FastISM(
            model, test_correctness=False)

        self.assertTrue(fast_ism_model.test_correctness())
    
    def test_conv_doub_res_mxp_two_fc(self):
        #               _________           _________
        #               ^       |           ^       |
        # inp -> C [ [ ->  C -> Add ] -> [ ->  C -> Add ] ] -> M -> [ D -> D -> y ]
        # doub_res_block contains 2 res_blocks within it -> double nesting
        doub_res = doub_res_block()
        fcs = fc_block(input_shape=(36*20,))

        inp = tf.keras.Input((108, 4))
        x = tf.keras.layers.Conv1D(20, 3, padding='same')(inp)
        x = doub_res(x)
        x = tf.keras.layers.MaxPooling1D(3)(x)
        x = tf.keras.layers.Flatten()(x)
        x = fcs(x)
        
        model = tf.keras.Model(inputs=inp, outputs=x)

        fast_ism_model = fastISM.FastISM(
            model, test_correctness=False)

        self.assertTrue(fast_ism_model.test_correctness())

if __name__ == '__main__':
    unittest.main()
