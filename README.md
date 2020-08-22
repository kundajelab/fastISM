# fastISM

[![PyPI version](https://img.shields.io/pypi/v/fastism.svg)](https://pypi.org/project/fastism/)

A Keras implementation for fast in-silico mutagenesis (ISM) for convolution-based architectures. It speeds up ISM by only restricting computation to those regions of each layer that are affected by a mutation in the input.

## Installation

Currently, fastISM is available to download from PyPI. Bioconda support is expected to be added in the future. fastISM requires TensorFlow 2.3.0 or above.
```bash
pip install fastism
```

## Usage

fastISM provides a simple interface that takes as input Keras model For any Keras ``model`` that takes in sequence as input of dimensions `(B, S, C)`, where
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

fastISM does a check for correctness when the model is initialised, which may take a few seconds depending on the size of your model. This ensures that the outputs of the model match that of an unoptimised implementation. You can turn it off as `FastISM(model, test_correctness=False)`. fastISM also supports models with multiple outputs.

## Benchmark
You can estimate the speedup obtained by comparing with a naive implementation of ISM.
```python
# Test this code as is
>>> from fastism import FastISM, NaiveISM
>>> from fastism.models.basset import basset_model
>>> import tensorflow as tf
>>> from time import time

>>> model = basset_model(seqlen=1000)
>>> naive_ism_model, fast_ism_model = NaiveISM(model), FastISM(model)

>>> def time_ism(m, x):
        t = time()
        m(x)
        return time()-t

>>> x = tf.random.uniform((1024, 1000, 4))

>>> time_ism(naive_ism_model, x)
144.013728
>>> time_ism(fast_ism_model, x)
13.894407
```
**TODO** Add benchmarking utilities. Test equality.

## Getting Help
fastISM supports the most commonly used subset of Keras for biological sequence-based models. Occasionally, you may find that some of the layers used in your model are not supported by fastISM (Supported Layers section in Documentation). In a few cases, the fastISM model may fail correctness checks, indicating there are likely some issues in the fastISM code. In both such cases or any other bugs, feel free to reach out to the author by posting an [Issue](https://github.com/kundajelab/fastISM/issues) on GitHub along with your architecture, and we'll try to work out a solution!

## Coming Soon
- Multi-input support
- Cropping1D support

## Citation
**TODO**