import tensorflow as tf
import unittest

from context import fastISM


class TestSimpleMultiInArchitectures(unittest.TestCase):
    def test_one_alt_inp_conv_add_error(self):
        # inp_seq -> C ->  Add -> D -> y
        # inp_alt -> C -----^
        #
        # Currently not supported to mix alternate with seq
        # before a STOP_LAYER. Should raise NotImplementedError
        inp_seq = tf.keras.Input((100, 4))
        inp_alt = tf.keras.Input((100, 4))
        x1 = tf.keras.layers.Conv1D(20, 3)(inp_seq)
        x2 = tf.keras.layers.Conv1D(20, 3)(inp_alt)
        x = tf.keras.layers.Add()([x1, x2])
        x = tf.keras.layers.Dense(1)(x)

        # both order of inputs
        model1 = tf.keras.Model(inputs=[inp_seq, inp_alt], outputs=x)
        model2 = tf.keras.Model(inputs=[inp_alt, inp_seq], outputs=x)

        with self.assertRaises(NotImplementedError):
            fastISM.FastISM(model1, seq_input_idx=0, test_correctness=False)

        with self.assertRaises(NotImplementedError):
            fastISM.FastISM(model2, seq_input_idx=0, test_correctness=False)

    def test_one_alt_inp_conv_cat_fc(self):
        # inp_seq -> C ->  Concat -> D -> y
        #            inp_alt --^
        inp_seq = tf.keras.Input((100, 4))
        inp_alt = tf.keras.Input((10,))
        x = tf.keras.layers.Conv1D(20, 3)(inp_seq)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Concatenate()([x, inp_alt])
        x = tf.keras.layers.Dense(1)(x)

        # both order of inputs
        model1 = tf.keras.Model(inputs=[inp_seq, inp_alt], outputs=x)
        model2 = tf.keras.Model(inputs=[inp_alt, inp_seq], outputs=x)

        fast_ism_model1 = fastISM.FastISM(
            model1, seq_input_idx=0, test_correctness=False)
        fast_ism_model2 = fastISM.FastISM(
            model2, seq_input_idx=1, test_correctness=False)

        self.assertTrue(fast_ism_model1.test_correctness())
        self.assertTrue(fast_ism_model2.test_correctness())

    def test_one_alt_inp_process_conv_cat_fc(self):
        # inp_seq -> C ->  Concat -> D -> y
        #       inp_alt -> D -^
        inp_seq = tf.keras.Input((100, 4))
        inp_alt = tf.keras.Input((10,))
        x = tf.keras.layers.Conv1D(20, 3)(inp_seq)
        x = tf.keras.layers.Flatten()(x)
        x_alt = tf.keras.layers.Dense(10)(inp_alt)
        x = tf.keras.layers.Concatenate()([x, x_alt])
        x = tf.keras.layers.Dense(1)(x)

        # both order of inputs
        model1 = tf.keras.Model(inputs=[inp_seq, inp_alt], outputs=x)
        model2 = tf.keras.Model(inputs=[inp_alt, inp_seq], outputs=x)

        fast_ism_model1 = fastISM.FastISM(
            model1, seq_input_idx=0, test_correctness=False)
        fast_ism_model2 = fastISM.FastISM(
            model2, seq_input_idx=1, test_correctness=False)

        self.assertTrue(fast_ism_model1.test_correctness())
        self.assertTrue(fast_ism_model2.test_correctness())

    def test_two_alt_inp_conv_cat_fc(self):
        #            inp_alt1 -|
        # inp_seq -> C ->  Concat -> D -> y
        #            inp_alt2 --^
        inp_seq = tf.keras.Input((100, 4))
        inp_alt1 = tf.keras.Input((10,))
        inp_alt2 = tf.keras.Input((10,))
        x = tf.keras.layers.Conv1D(20, 3)(inp_seq)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Concatenate()([x, inp_alt1, inp_alt2])
        x = tf.keras.layers.Dense(1)(x)

        # different order of inputs
        model1 = tf.keras.Model(
            inputs=[inp_seq, inp_alt1, inp_alt2], outputs=x)
        model2 = tf.keras.Model(
            inputs=[inp_alt2, inp_seq, inp_alt1], outputs=x)
        model3 = tf.keras.Model(
            inputs=[inp_alt2, inp_alt1, inp_seq], outputs=x)

        fast_ism_model1 = fastISM.FastISM(
            model1, seq_input_idx=0, test_correctness=False)
        fast_ism_model2 = fastISM.FastISM(
            model2, seq_input_idx=1, test_correctness=False)
        fast_ism_model3 = fastISM.FastISM(
            model3, seq_input_idx=2, test_correctness=False)

        self.assertTrue(fast_ism_model1.test_correctness())
        self.assertTrue(fast_ism_model2.test_correctness())
        self.assertTrue(fast_ism_model3.test_correctness())

    def test_two_alt_inp_conv_stagger(self):
        #            inp_alt1 -|
        # inp_seq -> C ->  Concat -> D -> Concat -> D -> y
        #                         inp_alt2 --^
        inp_seq = tf.keras.Input((100, 4))
        inp_alt1 = tf.keras.Input((10,))
        inp_alt2 = tf.keras.Input((10,))
        x = tf.keras.layers.Conv1D(20, 3)(inp_seq)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Concatenate()([x, inp_alt1])
        x = tf.keras.layers.Dense(10)(x)
        x = tf.keras.layers.Concatenate()([x, inp_alt2])
        x = tf.keras.layers.Dense(1)(x)

        # different order of inputs
        model1 = tf.keras.Model(
            inputs=[inp_seq, inp_alt1, inp_alt2], outputs=x)
        model2 = tf.keras.Model(
            inputs=[inp_alt2, inp_seq, inp_alt1], outputs=x)
        model3 = tf.keras.Model(
            inputs=[inp_alt2, inp_alt1, inp_seq], outputs=x)

        fast_ism_model1 = fastISM.FastISM(
            model1, seq_input_idx=0, test_correctness=False)
        fast_ism_model2 = fastISM.FastISM(
            model2, seq_input_idx=1, test_correctness=False)
        fast_ism_model3 = fastISM.FastISM(
            model3, seq_input_idx=2, test_correctness=False)

        self.assertTrue(fast_ism_model1.test_correctness())
        self.assertTrue(fast_ism_model2.test_correctness())
        self.assertTrue(fast_ism_model3.test_correctness())

    def test_two_alt_interact(self):
        # inp_seq -> C ->  Concat -> D -> y
        # inp_alt1 -> ADD ---^
        # inp_alt2 ----^
        inp_seq = tf.keras.Input((100, 4))
        inp_alt1 = tf.keras.Input((10,))
        inp_alt2 = tf.keras.Input((10,))
        inp_sum = tf.keras.layers.Add()([inp_alt1, inp_alt2])
        x = tf.keras.layers.Conv1D(20, 3)(inp_seq)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Concatenate()([x, inp_sum])
        x = tf.keras.layers.Dense(1)(x)

        # different order of inputs
        model1 = tf.keras.Model(
            inputs=[inp_seq, inp_alt1, inp_alt2], outputs=x)
        model2 = tf.keras.Model(
            inputs=[inp_alt2, inp_seq, inp_alt1], outputs=x)
        model3 = tf.keras.Model(
            inputs=[inp_alt2, inp_alt1, inp_seq], outputs=x)

        fast_ism_model1 = fastISM.FastISM(
            model1, seq_input_idx=0, test_correctness=False)
        fast_ism_model2 = fastISM.FastISM(
            model2, seq_input_idx=1, test_correctness=False)
        fast_ism_model3 = fastISM.FastISM(
            model3, seq_input_idx=2, test_correctness=False)

        self.assertTrue(fast_ism_model1.test_correctness())
        self.assertTrue(fast_ism_model2.test_correctness())
        self.assertTrue(fast_ism_model3.test_correctness())

    def test_one_alt_conv_cat_twice_fc(self):
        # inp_seq -> C ->  Concat -> D -> Concat -> D -> y
        #            inp_alt --^-----------^
        inp_seq = tf.keras.Input((100, 4))
        inp_alt = tf.keras.Input((10,))
        x = tf.keras.layers.Conv1D(20, 3)(inp_seq)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Concatenate()([x, inp_alt])
        x = tf.keras.layers.Dense(10)(x)
        x = tf.keras.layers.Concatenate()([inp_alt, x])
        x = tf.keras.layers.Dense(1)(x)

        # both order of inputs
        model1 = tf.keras.Model(inputs=[inp_seq, inp_alt], outputs=x)
        model2 = tf.keras.Model(inputs=[inp_alt, inp_seq], outputs=x)

        fast_ism_model1 = fastISM.FastISM(
            model1, seq_input_idx=0, test_correctness=False)
        fast_ism_model2 = fastISM.FastISM(
            model2, seq_input_idx=1, test_correctness=False)

        self.assertTrue(fast_ism_model1.test_correctness())
        self.assertTrue(fast_ism_model2.test_correctness())

    def test_two_alt_interact_complex(self):
        # inp_seq -> C ->  Concat -> D ---> Concat -> D -> y
        # inp_alt1 -> ADD ---^--|            ^
        # inp_alt2 ----^----> Concat --> D -->
        inp_seq = tf.keras.Input((100, 4))
        inp_alt1 = tf.keras.Input((10,))
        inp_alt2 = tf.keras.Input((10,))
        inp_sum = tf.keras.layers.Add()([inp_alt1, inp_alt2])
        x = tf.keras.layers.Conv1D(20, 3)(inp_seq)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Concatenate()([x, inp_sum])
        x = tf.keras.layers.Dense(20)(x)
        inp_alt2_sum = tf.keras.layers.Concatenate()([inp_sum, inp_alt2])
        inp_alt2_sum = tf.keras.layers.Dense(10)(inp_alt2_sum)
        x = tf.keras.layers.Concatenate()([inp_alt2_sum, x])
        x = tf.keras.layers.Dense(1)(x)

        # different order of inputs
        model1 = tf.keras.Model(
            inputs=[inp_seq, inp_alt1, inp_alt2], outputs=x)
        model2 = tf.keras.Model(
            inputs=[inp_alt2, inp_seq, inp_alt1], outputs=x)
        model3 = tf.keras.Model(
            inputs=[inp_alt2, inp_alt1, inp_seq], outputs=x)

        fast_ism_model1 = fastISM.FastISM(
            model1, seq_input_idx=0, test_correctness=False)
        fast_ism_model2 = fastISM.FastISM(
            model2, seq_input_idx=1, test_correctness=False)
        fast_ism_model3 = fastISM.FastISM(
            model3, seq_input_idx=2, test_correctness=False)

        self.assertTrue(fast_ism_model1.test_correctness())
        self.assertTrue(fast_ism_model2.test_correctness())
        self.assertTrue(fast_ism_model3.test_correctness())

    def test_one_alt_double_cat_three_out(self):
        # test multiple outputs
        #              |----> D -> y1
        # inp_seq -> C ->  Concat ----> D -> D -> y2
        #                     ^         |
        #       inp_alt -> D -|-> D ->Concat -> D -> y3
        inp_seq = tf.keras.Input((100, 4))
        inp_alt = tf.keras.Input((10,))
        x = tf.keras.layers.Conv1D(20, 3)(inp_seq)
        x = tf.keras.layers.Flatten()(x)
        y1 = tf.keras.layers.Dense(1)(x)
        x_alt = tf.keras.layers.Dense(10)(inp_alt)
        x = tf.keras.layers.Concatenate()([x, x_alt])
        x_alt = tf.keras.layers.Dense(10)(x_alt)
        x = tf.keras.layers.Dense(10)(x)
        x_alt = tf.keras.layers.Concatenate()([x, x_alt])
        y2 = tf.keras.layers.Dense(1)(x)
        y3 = tf.keras.layers.Dense(1)(x_alt)

        # both order of inputs
        model1 = tf.keras.Model(
            inputs=[inp_seq, inp_alt], outputs=[y1, y2, y3])
        model2 = tf.keras.Model(
            inputs=[inp_alt, inp_seq], outputs=[y1, y2, y3])

        fast_ism_model1 = fastISM.FastISM(
            model1, seq_input_idx=0, test_correctness=False)
        fast_ism_model2 = fastISM.FastISM(
            model2, seq_input_idx=1, test_correctness=False)

        self.assertTrue(fast_ism_model1.test_correctness())
        self.assertTrue(fast_ism_model2.test_correctness())

    def test_one_alt_double_cat_three_out_10bp_change_range(self):
        # test multiple outputs
        #              |----> D -> y1
        # inp_seq -> C ->  Concat ----> D -> D -> y2
        #                     ^         |
        #       inp_alt -> D -|-> D ->Concat -> D -> y3
        inp_seq = tf.keras.Input((100, 4))
        inp_alt = tf.keras.Input((10,))
        x = tf.keras.layers.Conv1D(20, 3)(inp_seq)
        x = tf.keras.layers.Flatten()(x)
        y1 = tf.keras.layers.Dense(1)(x)
        x_alt = tf.keras.layers.Dense(10)(inp_alt)
        x = tf.keras.layers.Concatenate()([x, x_alt])
        x_alt = tf.keras.layers.Dense(10)(x_alt)
        x = tf.keras.layers.Dense(10)(x)
        x_alt = tf.keras.layers.Concatenate()([x, x_alt])
        y2 = tf.keras.layers.Dense(1)(x)
        y3 = tf.keras.layers.Dense(1)(x_alt)

        # both order of inputs
        model1 = tf.keras.Model(
            inputs=[inp_seq, inp_alt], outputs=[y1, y2, y3])
        model2 = tf.keras.Model(
            inputs=[inp_alt, inp_seq], outputs=[y1, y2, y3])

        fast_ism_model1 = fastISM.FastISM(
            model1, seq_input_idx=0,
            change_ranges=[(i, i+10) for i in range(0, 100, 10)],
            test_correctness=False)
        fast_ism_model2 = fastISM.FastISM(
            model2, seq_input_idx=1,
            change_ranges=[(i, i+10) for i in range(0, 100, 10)],
            test_correctness=False)

        self.assertTrue(fast_ism_model1.test_correctness())
        self.assertTrue(fast_ism_model2.test_correctness())


if __name__ == '__main__':
    unittest.main()
