Examples
========

This section covers some of the common use cases and functionalities of fastISM.

fastISM provides a simple interface that takes as input Keras model For any Keras ``model`` that takes in sequence as input of dimensions ``(B, S, C)``, where 

- ``B``: batch size
- ``S``: sequence length
- ``C``: number of characters in vocabulary (e.g. 4 for DNA/RNA, 20 for proteins)

Alternate Mutations
-------------------
By default, inputs at the ith position are set to zero. It is possible to specify mutations of interest by passing them to ``replace_with`` in the call to the fastISM model. To perform ISM with all possible mutations for DNA:

.. code-block:: python

    fast_ism_model = FastISM(model)

    mutations = [[1,0,0,0],
                 [0,1,0,0],
                 [0,0,1,0],
                 [0,0,0,1]]

    for seq_batch in sequences:
        # seq_batch has dim (B, S, C)
        for m in mutations:
            ism_seq_batch = fast_ism_model(seq_batch, replace_with=m)
            # ism_seq_batch has dim (B, S, num_outputs) 
            # process/store ism_seq_batch

Each ``ism_seq_batch`` has the same dimensions ``(B, S, num_outputs)``. The outputs of the model are computed on the mutations only for those positions where the base differs from the mutation. Where the base is the same as the mutation, the output is the same as for the unperturbed sequence.

Alternate Ranges
----------------
By default, mutations are introduced at every single position in the input. You can also set a list of equal-sized ranges as input instead of single positions. Consider a model that takes as input 1000 length sequences, and we wish to introduce a specific mutation of length 3 in the central 150 positions:

**TODO**: test this

.. code-block:: python

    # specific mutation to introduce
    mut = [[0,0,0,1],
           [0,0,0,1],
           [0,0,0,1]]
    
    # ranges where mutation should be introduced
    mut_ranges = [(i,i+3) for i in range(425,575)]
    
    fast_ism_model = FastISM(model, 
                             change_ranges = mut_ranges)

    for seq_batch in sequences:
        ism_seq_batch = fast_ism_model(seq_batch, replace_with=mut)   

Multi-input Models
------------------
fastISM supports models which have other inputs in addition to the sequence input that is perturbed. These alternate inputs are assumed to stay constant through different perturbations of the primary sequence input. Consider the model below in which an addition vector is concatenated with the flattened sequence output:

.. code-block:: python

   def get_model():
       rna = tf.keras.Input((100,)) # non-sequence input
       seq = tf.keras.Input((100,4))
     
       x = tf.keras.layers.Conv1D(20, 3)(seq)
       x = tf.keras.layers.Conv1D(20, 3)(x)
       x = tf.keras.layers.Flatten()(x)
 
       rna_fc = tf.keras.layers.Dense(10)(rna)
 
       x = tf.keras.layers.Concatenate()([x, rna_fc])
       x = tf.keras.layers.Dense(10)(x)
       x = tf.keras.layers.Dense(1)(x)
       model = tf.keras.Model(inputs=[rna,seq], outputs=x)
       
       return model

To inform fastISM that the second input is the primary sequence input that will be perturbed:

.. code-block:: python

   >>> model = get_model()
   >>> fast_ism_model = FastISM(model, seq_input_idx=1) 

Then to obtain the outputs:

.. code-block:: python

   for rna_batch, seq_batch in data_batches:
      ism_batch = fast_ism_model([rna_batch, seq_batch]) 
 
   # or equivalently without splitting inputs
   for data_batch in data_batches
       ism_batch = fast_ism_model(data_batch)

**NOTE**: Currently, multi-input models in which descendants of alternate inputs interact directly with descendants of primary sequence input *before* a :ref:`Stop Layer <stop-layers>` are not supported, i.e. a descendant of an alternate input in general should only interact with a flattened version of primary input sequence.

Recursively Defined Models
--------------------------
Keras allows defining models in a nested fashion. As such, recursively defined models should not pose an issue to fastISM. The example below works:

.. code-block:: python

   def res_block(input_shape):
       inp = tf.keras.Input(shape=input_shape)
       x = tf.keras.layers.Conv1D(20, 3, padding='same')(inp)    
       x = tf.keras.layers.Add()([inp, x])
       model = tf.keras.Model(inputs=inp, outputs=x)
       return model
  
   def fc_block(input_shape):
       inp = tf.keras.Input(shape=input_shape)
       x = tf.keras.layers.Dense(10)(inp)
       x = tf.keras.layers.Dense(1)(x)
     
       model = tf.keras.Model(inputs=inp, outputs=x)
       return model
 
   def get_model():
       res = res_block(input_shape=(108,20)))
       fcs = fc_block(input_shape=(36*20,))
 
       inp = tf.keras.Input((108, 4))
       x = tf.keras.layers.Conv1D(20, 3, padding='same')(inp)
       x = res(x)
       x = tf.keras.layers.MaxPooling1D(3)(x)
       x = tf.keras.layers.Flatten()(x)
       x = fcs(x)
 
       model = tf.keras.Model(inputs=inp, outputs=x)
     
       return model
 
   >>> model = get_model()
   >>> fast_ism_model = FastISM(model)