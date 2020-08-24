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

def my_add_block(input_shape=(108,20)):
    x1 = tf.keras.Input(shape=input_shape)
    x2 = tf.keras.Input(shape=input_shape)
    y = tf.keras.layers.Add()([x1,x2])
    
    model = tf.keras.Model(inputs=[x1,x2], outputs=y)
    return model

def my_add_max_block(input_shape=(108,20)):
    x1 = tf.keras.Input(shape=input_shape)
    x2 = tf.keras.Input(shape=input_shape)
    y1 = tf.keras.layers.Add()([x1,x2])
    y2 = tf.keras.layers.Maximum()([x1,x2])
    
    model = tf.keras.Model(inputs=[x1,x2], outputs=[y1, y2])
    return model


def my_sub_block(input_shape=(108,20)):
    x1 = tf.keras.Input(shape=input_shape)
    x2 = tf.keras.Input(shape=input_shape)
    y = tf.keras.layers.Subtract()([x2,x1])
    
    model = tf.keras.Model(inputs=[x1,x2], outputs=y)
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
    
    def test_conv_my_add_mxp_two_fc(self):
        #            _________
        #            ^       |
        # inp -> C  ->  C ->[ Add ] -> M -> [ D -> D -> y ]
        # testing a nested block that takes in multiple inputs
        my_add = my_add_block()
        fcs = fc_block(input_shape=(36*20,))

        inp = tf.keras.Input((108, 4))
        x1 = tf.keras.layers.Conv1D(20, 3, padding='same')(inp)
        x2 = tf.keras.layers.Conv1D(20, 3, padding='same')(x1)
        y = my_add([x1,x2])
        y = tf.keras.layers.MaxPooling1D(3)(y)
        y = tf.keras.layers.Flatten()(y)
        y = fcs(y)
        
        model = tf.keras.Model(inputs=inp, outputs=y)

        fast_ism_model = fastISM.FastISM(
            model, test_correctness=False)

        self.assertTrue(fast_ism_model.test_correctness())
    
    def test_conv_my_add_max_mxp_two_fc(self):
        #            _________     __________
        #            ^       |     ^        |
        # inp -> C  ->  C ->[ Add/Max ] -> Add -> M -> [ D -> D -> y ]
        # testing a nested block that takes in multiple inputs
        # and returns  multiple outputs
        my_add_max = my_add_max_block()
        fcs = fc_block(input_shape=(36*20,))

        inp = tf.keras.Input((108, 4))
        x1 = tf.keras.layers.Conv1D(20, 3, padding='same')(inp)
        x2 = tf.keras.layers.Conv1D(20, 3, padding='same')(x1)
        y1, y2 = my_add_max([x1,x2])
        y = tf.keras.layers.Add()([y1,y2])
        y = tf.keras.layers.MaxPooling1D(3)(y)
        y = tf.keras.layers.Flatten()(y)
        y = fcs(y)
        
        model = tf.keras.Model(inputs=inp, outputs=y)

        fast_ism_model = fastISM.FastISM(
            model, test_correctness=False)

        self.assertTrue(fast_ism_model.test_correctness())
    
    def test_conv_my_sub_mxp_two_fc(self):
        # TODO: fails as of now since inbound_edges does not contain
        # the correct node order
        #            _________
        #            ^       |
        # inp -> C  ->  C ->[ Sub ] -> M -> [ D -> D -> y ]
        # testing a nested block that takes in multiple inputs
        my_sub = my_sub_block()
        fcs = fc_block(input_shape=(36*20,))

        inp = tf.keras.Input((108, 4))
        x1 = tf.keras.layers.Conv1D(20, 3, padding='same')(inp)
        x2 = tf.keras.layers.Conv1D(20, 3, padding='same')(x1)
        y = my_sub([x1,x2])
        y = tf.keras.layers.MaxPooling1D(3)(y)
        y = tf.keras.layers.Flatten()(y)
        y = fcs(y)
        
        model = tf.keras.Model(inputs=inp, outputs=y)

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
