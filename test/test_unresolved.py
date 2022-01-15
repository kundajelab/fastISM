import tensorflow as tf
import unittest

from context import fastISM

class TestUnresolved(unittest.TestCase):
    """
    These are outstanding issues that need to be solved 
    with short descriptions of the problem and possible
    solutions.
    """ 

    def test_stop_multi_input(self):
        """
        In this case the stop segment has different inputs
        at different nodes, and this confuses the change range
        computation step and breaks assertions.

        One way to fix it could be to modify the segmenting code
        such that nodes that are encountered again and are already
        in stop segment should be updated with a new idx, and
        in the label_stop_descendants step nodes that are already
        labeled with (potentially non-stop) index are given a new 
        index. This would likely fail if a stop layer gets non-stop
        inputs of different widths.
        """
        inp = tf.keras.Input((100, 4))
        x = tf.keras.layers.Conv1D(10, 3, padding='same')(inp)
        y = tf.keras.layers.Dense(10)(x)
        x = tf.keras.layers.Add()([x,y])
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs=inp, outputs=x)

        fast_ism_model = fastISM.FastISM(
            model, test_correctness=False)

        self.assertTrue(fast_ism_model.test_correctness())

