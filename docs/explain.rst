How fastISM Works!
==================

This section gives a high level overview of the fastISM algorithm. For more detail, check out the paper, or better still, take a look at the `source code <https://github.com/kundajelab/fastISM>`_!

.. figure:: ../images/annotated_basset.pdf
   
Annotated diagram of a Basset-like architecture `(Kelley et al., 2016) <https://pubmed.ncbi.nlm.nih.gov/27197224/>`_ on an input DNA sequence of length 1000, with a 1 base-pair mutation at position 500. Positions marked in red indicate the regions that are affected by the point mutation in the input. Positions marked in yellow, flanking the positions in red, indicate unaffected regions that contribute to the output of the next layer. Ticks at the bottom of each layer correspond to position indices. Numbers on the right in black indicate the approximate number of computations required at that layer for a naive implementation of ISM. For convolution layers, the numbers in gray and green indicate the minimal computations required.
