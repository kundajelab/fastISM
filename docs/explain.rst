How fastISM Works
=================

This section gives a high level overview of the fastISM algorithm. For more detail, check out the paper, or better still, take a look at the `source code <https://github.com/kundajelab/fastISM>`_!

fastISM is based on the observation that neural networks spend the majority of their computation time in convolutional layers and that point mutations in the input sequence only affect limited a range of intermediate layers. As a result, most of the computation in ISM is redundant and avoiding it can result in significant speedups.

.. figure:: ../images/annotated_basset.pdf
   
Consider the above annotated diagram of a Basset-like architecture `(Kelley et al., 2016) <https://pubmed.ncbi.nlm.nih.gov/27197224/>`_ on an input DNA sequence of length 1000, with a 1 base-pair mutation at position 500. Positions marked in red indicate the regions that are affected by the point mutation in the input. Positions marked in yellow, flanking the positions in red, indicate unaffected regions that contribute to the output of the next layer. Ticks at the bottom of each layer correspond to position indices. Numbers on the right in black indicate the approximate number of computations required at that layer for a naive implementation of ISM. For convolution layers, the numbers in gray and green indicate the minimal computations required.

For a single position change in the middle of the input sequence, the output of the first convolution, which has a kernel size of 19, is perturbed at 19 positions which can be computed from just 37 positions in the input. It then goes on to affect 7 out of 333 positions after the first Max Pool layer (Layer 2) and 5 out of 83 positions after the second Max Pool (Layer 3). Once the output of the final Max Pool layer is flattened and passed through a fully connected layer, all the neurons are affected by a single change in the input, and thus all subsequent computations must be recomputed entirely.

fastISM works by restricting computation in the convolution layers to only those positions that are affected by the mutation in the input. Since the most time is spent in convolution layers, fastISM avoids down a major amount of redundant computation and speeds up ISM. See :ref:`API <api>` for more details on how this is achieved.