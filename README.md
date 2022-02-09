[![](https://github.com/kundajelab/fastISM/raw/docs/images/logo.jpeg)](https://github.com/kundajelab/fastISM)

[![](https://img.shields.io/pypi/v/fastism.svg)](https://pypi.org/project/fastism/) [![](https://readthedocs.org/projects/fastism/badge/?version=latest)](https://fastism.readthedocs.io/en/latest/?badge=latest)

# Quickstart

A Keras implementation for fast in-silico saturated mutagenesis (ISM) for convolution-based architectures. It speeds up ISM by 10x or more by restricting computation to those regions of each layer that are affected by a mutation in the input.

## Installation

Currently, fastISM is available to download from PyPI. Bioconda support is expected to be added in the future. fastISM requires TensorFlow 2.3.0 or above.
```bash
pip install fastism
```

## Usage

fastISM provides a simple interface that takes as input Keras models. For any Keras ``model`` that takes in sequence as input of dimensions `(B, S, C)`, where
- `B`: batch size
- `S`: sequence length
- `C`: number of characters in vocabulary (e.g. 4 for DNA/RNA, 20 for proteins)

Perform ISM as follows:

```python
from fastism import FastISM

fast_ism_model = FastISM(model)

for seq_batch in sequences:
    # seq_batch has dim (B, S, C)
    ism_seq_batch = fast_ism_model(seq_batch)
    # ism_seq_batch has dim (B, S, num_outputs) 
```

fastISM does a check for correctness when the model is initialised, which may take a few seconds depending on the size of your model. This ensures that the outputs of the model match that of an unoptimised implementation. You can turn it off as `FastISM(model, test_correctness=False)`. fastISM also supports introducing specific mutations, mutating different ranges of the input sequence, and models with multiple outputs. Check the [Examples](https://fastism.readthedocs.io/en/latest/examples.html) section of the documentation for more details. An executable tutorial is available on [Colab](https://colab.research.google.com/github/kundajelab/fastISM/blob/master/notebooks/colab/DeepSEA.ipynb).

## Benchmark
You can estimate the speedup obtained by comparing with a naive implementation of ISM.
```python
# Test this code as is
>>> from fastism import FastISM, NaiveISM
>>> from fastism.models.basset import basset_model
>>> import tensorflow as tf
>>> import numpy as np
>>> from time import time

>>> model = basset_model(seqlen=1000)
>>> naive_ism_model = NaiveISM(model)
>>> fast_ism_model = FastISM(model)

>>> def time_ism(m, x):
        t = time()
        o = m(x)
        print(time()-t)
        return o

>>> x = tf.random.uniform((1024, 1000, 4),
                          dtype=model.input.dtype)

>>> naive_out = time_ism(naive_ism_model, x)
144.013728
>>> fast_out = time_ism(fast_ism_model, x)
13.894407
>>> np.allclose(naive_out, fast_out, atol=1e-6) 
True
>>> np.allclose(fast_out, naive_out, atol=1e-6) 
True # np.allclose is not symmetric
```

See `notebooks/ISMBenchmark.ipynb` for benchmarking code that accounts for initial warm-up.

## Getting Help
fastISM supports the most commonly used subset of Keras for biological sequence-based models. Occasionally, you may find that some of the layers used in your model are not supported by fastISM. Refer to the [Supported Layers](https://fastism.readthedocs.io/en/latest/layers.html) section in Documentation for instructions on how to incorporate custom layers. In a few cases, the fastISM model may fail correctness checks, indicating there are likely some issues in the fastISM code. In such cases or any other bugs, feel free to reach out to the author by posting an [Issue](https://github.com/kundajelab/fastISM/issues) on GitHub along with your architecture, and we'll try to work out a solution!

## Citation
fastISM: Performant *in-silico* saturation mutagenesis for convolutional neural networks; Surag Nair, Avanti Shrikumar, Anshul Kundaje (bioRxiv 2020)
[https://doi.org/10.1101/2020.10.13.337147](https://doi.org/10.1101/2020.10.13.337147)

