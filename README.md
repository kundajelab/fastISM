# FastISM

[![PyPI version](https://badge.fury.io/py/fastism.svg)](https://badge.fury.io/py/fastism) [![Build Status](https://travis-ci.com/kundajelab/fastISM.svg?token=AGJ26SY8T83y31vHU39u&branch=master)](https://travis-ci.com/kundajelab/fastISM)

A Keras implementation for fast in-silico mutagenesis for convolution-based architectures.

## Installation

```bash
pip install fastism
```

## Usage

For any Keras `model` that takes in sequence as input of dimensions `(batch_size, seqlen, num_chars)`, perform ISM as follows:

```python
from fastism import FastISM

fast_ism_model = FastISM(model)

for seq_batch in sequences:
    ism_seq_batch = fast_ism_model(seq_batch)
```
