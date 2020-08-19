from .ism_base import ISMBase, NaiveISM
from .fast_ism_utils import generate_models

import tensorflow as tf
import numpy as np


class FastISM(ISMBase):
    def __init__(self, model, seq_input_idx=0, change_ranges=None, replace_with=0, test_correctness=True):
        super().__init__(model, seq_input_idx, change_ranges, replace_with)

        self.intermediate_output_model, self.intout_output_tensors, \
            self.fast_ism_model, self.input_specs = generate_models(
                self.model, self.seqlen, self.num_chars, self.seq_input_idx, self.change_ranges)

        self.intout_output_tensor_to_idx = {
            x: i for i, x in enumerate(self.intout_output_tensors)}

        if test_correctness:
            if not self.test_correctness():
                raise ValueError("Fast ISM model built is incorrect, likely \
                                 due to internal error. Please post an Issue \
                                      with your architecture.")                

    def __call__(self, seq_batch):
        # run intermediate output on unperturbed sequence
        intout_output = self.intermediate_output_model(seq_batch)
        padded_inputs = self.prepare_intout_output(
            intout_output, seq_batch.shape[0])  # better name than padded?

        ism_outputs = []
        for i in range(len(self.change_ranges)):
            fast_ism_inputs = self.prepare_ith_input(padded_inputs, i)

            # TODO: what if multiple outputs
            ism_output = self.fast_ism_model(fast_ism_inputs).numpy()
            ism_outputs.append(ism_output)

        # seq x num_mut x output_dim
        ism_outputs = np.swapaxes(np.array(ism_outputs), 0, 1)
        return ism_outputs

    def prepare_intout_output(self, intout_output, num_seqs):
        inputs = []

        for input_spec in self.input_specs:
            if input_spec[0] == "SEQ_PERTURB":
                inputs.append(tf.tile(self.perturbation, [num_seqs, 1, 1]))
            elif input_spec[0] == "INTOUT":
                # pad the output if required
                to_pad = intout_output[self.intout_output_tensor_to_idx[input_spec[1]['node']]]
                num_filts = to_pad.shape[2]

                padded = tf.concat([
                    # left zeros
                    tf.zeros(
                        (num_seqs, input_spec[1]['padding'][0], num_filts)),
                    to_pad,
                    # right zeros
                    tf.zeros(
                        (num_seqs, input_spec[1]['padding'][1], num_filts)),
                ], axis=1)
                inputs.append(padded)

            elif input_spec[0] == "OFFSET":
                # nothing for now, add i specific offset later
                inputs.append(None)
            else:
                raise ValueError(
                    "{}: what is this input spec?".format(input_spec[0]))

        return inputs

    def prepare_ith_input(self, padded_inputs, i):
        inputs = []

        for input_idx, input_spec in enumerate(self.input_specs):
            if input_spec[0] == "SEQ_PERTURB":
                inputs.append(padded_inputs[input_idx])
            elif input_spec[0] == "INTOUT":
                # slice
                inputs.append(
                    padded_inputs[input_idx][:,
                                             input_spec[1]['slices'][i][0]: input_spec[1]['slices'][i][1]])
            elif input_spec[0] == "OFFSET":
                inputs.append(input_spec[1]['offsets'][i])
            else:
                raise ValueError(
                    "{}: what is this input spec?".format(input_spec[0]))

        return inputs

    def test_correctness(self, batch_size=10, atol=1e-6):
        """
        Verify that outputs are correct by matching with Naive ISM. Running on small
        examples so as to not take too long. 

        Hence not comparing runtime against Naive ISM implementation, which requires
        bigger inputs to offset overheads.

        TODO: ensure generated data is on GPU already before calling either method (for speedup)
        """

        # TODO: better way to do this?
        naive_ism = NaiveISM(self.model, self.seq_input_idx,
                             self.change_ranges, self.replace_with)

        # test batch
        x = tf.constant(np.random.random(
            (batch_size, self.seqlen, self.num_chars)))

        naive_out = naive_ism(x)
        fast_out = self(x)

        return np.all(np.isclose(naive_out, fast_out, atol=atol))

    def time_batch(self, seq_batch):
        pass
