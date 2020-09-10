fastISM package
===============

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

