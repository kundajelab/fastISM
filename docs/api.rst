.. _api:

fastISM package
===============

fastISM takes a Keras model as input. The main steps of fastISM are as follows:

1. One-time Initialization (:py:func:`fastISM.fast_ism_utils.generate_models`):

 - Obtain the computational graph from the model. This is done in :py:func:`fastISM.flatten_model.get_flattened_graph`.
 - Chunk the computational graph into segments that can be run as a unit. This is done in :py:func:`fastISM.fast_ism_utils.segment_model`.
 - Augment the model to create an “intermediate output model” (referred to as ``intout_model`` in the code) that returns intermediate outputs at the end of each segment for reference input sequences. This is done in :py:func:`fastISM.fast_ism_utils.generate_intermediate_output_model`.
 - Create a second “mutation propagation model” (referred to as ``fast_ism_model`` in the code) that largely resembles the original model, but incorporates as additional inputs the necessary flanking regions from outputs of the IntOut model on reference input sequences between segments. This is done in :py:func:`fastISM.fast_ism_utils.generate_fast_ism_model`.


2. For each batch of input sequences:

 - Run the ``intout_model`` on the sequences (unperturbed) and cache the intermediate outputs at the end of each segment. This is done in :py:func:`fastISM.fast_ism.FastISM.pre_change_range_loop_prep`.
 - For each positional mutation:

   - Introduce the mutation in the input sequences
   - Run the ``fast_ism_model`` feeding as input appropriate slices of the ``intout_model`` outputs. This is done in :py:func:`fastISM.fast_ism.FastISM.get_ith_output`.

See :ref:`How fastISM Works <explain>` for a more intuitive understanding of the algorithm.

ism\_base module
------------------------

This module contains a :class:`base ISM <fastISM.ism_base.ISMBase>` class, from which the :class:`NaiveISM <fastISM.ism_base.NaiveISM>` and :class:`FastISM <fastISM.fast_ism.FastISM>` classes inherit. It also includes implementation of :class:`NaiveISM <fastISM.ism_base.NaiveISM>`.

.. automodule:: fastISM.ism_base
   :members:
   :undoc-members:
   :show-inheritance:

fast\_ism module
------------------------

This module contains the :class:`FastISM <fastISM.fast_ism.FastISM>` class.

.. automodule:: fastISM.fast_ism
   :members:
   :undoc-members:
   :show-inheritance:

fast\_ism\_utils module
-------------------------------

.. automodule:: fastISM.fast_ism_utils
   :members:
   :undoc-members:
   :show-inheritance:

change\_range module
----------------------------

.. automodule:: fastISM.change_range
   :members:
   :undoc-members:
   :show-inheritance:


flatten\_model module
-----------------------------

This module implements functions required to take an arbitrary Keras model and reduce them to a graph representation that is then manipulated by :mod:`fast_ism_utils <fastISM.fast_ism_utils>`.

.. automodule:: fastISM.flatten_model
   :members:
   :undoc-members:
   :show-inheritance:

