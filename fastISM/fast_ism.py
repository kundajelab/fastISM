from .ism_base import ISMBase, NaiveISM
from .fast_ism_utils import generate_models

import tensorflow as tf
import numpy as np


class FastISM(ISMBase):
    def __init__(self, model, seq_input_idx=0, change_ranges=None,
                 early_stop_layers=None, test_correctness=True):
        super().__init__(model, seq_input_idx, change_ranges)

        self.output_nodes, self.intermediate_output_model, self.intout_output_tensors, \
            self.fast_ism_model, self.input_specs = generate_models(
                self.model, self.seqlen, self.num_chars, self.seq_input_idx,
                self.change_ranges, early_stop_layers)

        self.intout_output_tensor_to_idx = {
            x: i for i, x in enumerate(self.intout_output_tensors)}

        if test_correctness:
            if not self.test_correctness():
                raise ValueError("""Fast ISM model built is incorrect, likely 
                                 due to internal errors. Please post an Issue 
                                 with your architecture and the authors will
                                 try their best to help you!""")

    def pre_change_range_loop_prep(self, inp_batch, num_seqs):
        # run intermediate output on unperturbed sequence
        intout_output = self.intermediate_output_model(
            inp_batch, training=False)

        self.padded_inputs = self.prepare_intout_output(
            intout_output, num_seqs)  # better name than padded?

        # return output on unperturbed input
        if self.num_outputs == 1:
            return intout_output[self.intout_output_tensor_to_idx[self.output_nodes[0]]]
        else:
            return [intout_output[self.intout_output_tensor_to_idx[self.output_nodes[i]]] for
                    i in range(self.num_outputs)]

    def prepare_intout_output(self, intout_output, num_seqs):
        inputs = []

        for input_spec in self.input_specs:
            if input_spec[0] == "SEQ_PERTURB":
                inputs.append(tf.tile(self.perturbation, [num_seqs, 1, 1]))
            elif input_spec[0] == "INTOUT_SEQ":
                # pad the output if required
                to_pad = intout_output[self.intout_output_tensor_to_idx[input_spec[1]['node']]]
                padded = tf.keras.layers.ZeroPadding1D(
                    input_spec[1]['padding'])(to_pad)
                inputs.append(padded)
            elif input_spec[0] == "INTOUT_ALT":
                # descendant of alternate input -- copy through
                inputs.append(
                    intout_output[self.intout_output_tensor_to_idx[input_spec[1]['node']]])
            elif input_spec[0] == "OFFSET":
                # nothing for now, add i specific offset later
                inputs.append(None)
            else:
                raise ValueError(
                    "{}: what is this input spec?".format(input_spec[0]))

        return inputs

    def get_ith_output(self, inp_batch, i, idxs_to_mutate):
        fast_ism_inputs = self.prepare_ith_input(
            self.padded_inputs, i, idxs_to_mutate)

        return self.fast_ism_model(fast_ism_inputs, training=False)

    def prepare_ith_input(self, padded_inputs, i, idxs_to_mutate):
        num_to_mutate = idxs_to_mutate.shape[0]
        inputs = []

        for input_idx, input_spec in enumerate(self.input_specs):
            if input_spec[0] == "SEQ_PERTURB":
                inputs.append(padded_inputs[input_idx][:num_to_mutate])
            elif input_spec[0] == "INTOUT_SEQ":
                # slice
                inputs.append(
                    tf.gather(padded_inputs[input_idx][:,
                                                       input_spec[1]['slices'][i][0]: input_spec[1]['slices'][i][1]],
                              idxs_to_mutate))
            elif input_spec[0] == "INTOUT_ALT":
                inputs.append(
                    tf.gather(padded_inputs[input_idx], idxs_to_mutate))
            elif input_spec[0] == "OFFSET":
                inputs.append(input_spec[1]['offsets'][i])
            else:
                raise ValueError(
                    "{}: what is this input spec?".format(input_spec[0]))

        return inputs

    def cleanup(self):
        # padded inputs no longer required, take up GPU memory
        # if not deleted, have led to memory leaks
        del self.padded_inputs

    def test_correctness(self, batch_size=10, replace_with=0, atol=1e-6):
        """
        Verify that outputs are correct by matching with Naive ISM. Running on small
        examples so as to not take too long. 

        Hence not comparing runtime against Naive ISM implementation, which requires
        bigger inputs to offset overheads.

        TODO: ensure generated data is on GPU already before calling either method (for speedup)
        """

        # TODO: better way to do this?
        naive_ism = NaiveISM(self.model, self.seq_input_idx,
                             self.change_ranges)

        # test batch
        if self.num_inputs == 1:
            x = tf.constant(np.random.random(
                (batch_size,) + self.model.input_shape[1:]),
                dtype=self.model.inputs[self.seq_input_idx].dtype)
        else:
            x = []
            for j in range(self.num_inputs):
                x.append(
                    tf.constant(np.random.random(
                        (batch_size,) + self.model.input_shape[j][1:]),
                        dtype=self.model.inputs[j].dtype)
                )

        naive_out = naive_ism(x, replace_with=replace_with)
        fast_out = self(x, replace_with=replace_with)

        if self.num_outputs == 1:
            return np.all(np.isclose(naive_out, fast_out, atol=atol))
        else:
            return all([np.allclose(naive_out[j], fast_out[j], atol=atol) and
                        np.allclose(fast_out[j], naive_out[j], atol=atol) for
                        j in range(self.num_outputs)])

    def time_batch(self, seq_batch):
        pass
