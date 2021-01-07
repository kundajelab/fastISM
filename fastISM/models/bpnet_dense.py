import tensorflow as tf
from tensorflow.keras.backend import int_shape
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, \
    Reshape, Input, Concatenate, Cropping1D, Add, Dropout, Reshape, Dense, \
    Activation, Flatten, Conv1D, GlobalMaxPooling1D, MaxPooling1D, \
    GlobalAveragePooling1D, BatchNormalization
import argparse


def get_model_param_dict(param_file):
    ''' 
    param_file has 2 columns -- param name in column 1, and param value in column 2 
    '''
    params = {}
    if param_file is None:
        return params
    for line in open(param_file, 'r').read().strip().split('\n'):
        tokens = line.split('\t')
        params[tokens[0]] = tokens[1]
    return params


def bpnet_dense_model(inlen=1346, outlen=1000, filters=32, ndl=6):
    """
    Anna's BPNet architecture based on
    https://github.com/kundajelab/kerasAC/blob/1b483c4/kerasAC/architectures/profile_bpnet_chipseq.py
    """
    filters = filters
    n_dil_layers = ndl
    conv1_kernel_size = 21
    profile_kernel_size = 75
    counts_loss_weight = 1
    profile_loss_weight = 1

    parser = argparse.ArgumentParser(description="view model arch")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--init_weights", default=None)
    parser.add_argument("--tdb_input_flank", nargs="+", default=[str(inlen//2)])
    parser.add_argument("--tdb_output_flank", nargs="+", default=[str(outlen//2)])
    parser.add_argument("--num_tasks", type=int, default=2)
    parser.add_argument("--model_params", default=None)

    args = parser.parse_args("")

    model_params = get_model_param_dict(args.model_params)
    if 'filters' in model_params:
        filters = int(model_params['filters'])
    if 'n_dil_layers' in model_params:
        n_dil_layers = int(model_params['n_dil_layers'])
    if 'conv1_kernel_size' in model_params:
        conv1_kernel_size = int(model_params['conv1_kernel_size'])
    if 'profile_kernel_size' in model_params:
        profile_kernel_size = int(model_params['profile_kernel_size'])
    if 'counts_loss_weight' in model_params:
        counts_loss_weight = float(model_params['counts_loss_weight'])
    if 'profile_loss_weight' in model_params:
        profile_loss_weight = float(model_params['profile_loss_weight'])

    #read in arguments
    seed = args.seed
    init_weights = args.init_weights
    sequence_flank = int(args.tdb_input_flank[0].split(',')[0])
    num_tasks = args.num_tasks

    seq_len = 2*sequence_flank
    out_flank = int(args.tdb_output_flank[0].split(',')[0])
    out_pred_len = 2*out_flank

    # define inputs
    inp = Input(shape=(seq_len, 4), name='sequence')
    bias_counts_input = Input(shape=(num_tasks,), name='control_logcount')
    bias_profile_input = Input(
        shape=(out_pred_len, num_tasks), name='control_profile')
    # first convolution without dilation
    first_conv = Conv1D(filters,
                        kernel_size=conv1_kernel_size,
                        padding='valid',
                        activation='relu',
                        name='1st_conv')(inp)
    # 6 dilated convolutions with resnet-style additions
    # each layer receives the sum of feature maps
    # from all previous layers
    res_layers = [(first_conv, '1stconv')]  # on a quest to have meaninful
    # layer names
    layer_names = [str(i)+"_dil" for i in range(n_dil_layers)]
    for i in range(1, n_dil_layers + 1):
        if i == 1:
            res_layers_sum = first_conv
        else:
            res_layers_sum = Add(name='add_{}'.format(i))(
                [l for l, _ in res_layers])

        # dilated convolution
        conv_layer_name = '{}conv'.format(layer_names[i-1])
        conv_output = Conv1D(filters,
                             kernel_size=3,
                             padding='valid',
                             activation='relu',
                             dilation_rate=2**i,
                             name=conv_layer_name)(res_layers_sum)

        # get shape of latest layer and crop
        # all other previous layers in the list to that size
        conv_output_shape = int_shape(conv_output)
        cropped_layers = []
        for lyr, name in res_layers:
            lyr_shape = int_shape(lyr)
            cropsize = int(lyr_shape[1]/2) - int(conv_output_shape[1]/2)
            lyr_name = '{}-crop_{}th_dconv'.format(name.split('-')[0], i)
            cropped_layers.append((Cropping1D(cropsize,
                                              name=lyr_name)(lyr),
                                   lyr_name))

        # append to the list of previous layers
        cropped_layers.append((conv_output, conv_layer_name))
        res_layers = cropped_layers

    # the final output from the 6 dilated convolutions
    # with resnet-style connections
    combined_conv = Add(name='combined_conv')([l for l, _ in res_layers])

    # Branch 1. Profile prediction
    # Step 1.1 - 1D convolution with a very large kernel
    profile_out_prebias = Conv1D(filters=num_tasks,
                                 kernel_size=profile_kernel_size,
                                 padding='valid',
                                 name='profile_out_prebias')(combined_conv)
    # Step 1.2 - Crop to match size of the required output size, a minimum
    #            difference of 346 is required between input seq len and ouput len
    profile_out_prebias_shape = int_shape(profile_out_prebias)
    cropsize = int(profile_out_prebias_shape[1]/2)-int(out_pred_len/2)
    profile_out_prebias = Cropping1D(cropsize,
                                     name='prof_out_crop2match_output')(profile_out_prebias)
    # Step 1.3 - concatenate with the control profile
    concat_pop_bpi = Concatenate(axis=-1, name='concat_with_bias_prof')([profile_out_prebias,
                                                                         bias_profile_input])

    # Step 1.4 - Final 1x1 convolution
    profile_out = Conv1D(filters=num_tasks,
                         kernel_size=1,
                         name="profile_predictions")(concat_pop_bpi)
    # Branch 2. Counts prediction
    # Step 2.1 - Global average pooling along the "length", the result
    #            size is same as "filters" parameter to the BPNet function
    gap_combined_conv = GlobalAveragePooling1D(
        name='gap')(combined_conv)  # acronym - gapcc

    # Step 2.2 Concatenate the output of GAP with bias counts
    concat_gapcc_bci = Concatenate(
        name="concat_with_bias_cnts", axis=-1)([gap_combined_conv, bias_counts_input])

    # Step 2.3 Dense layer to predict final counts
    count_out = Dense(num_tasks, name="logcount_predictions")(concat_gapcc_bci)

    # instantiate keras Model with inputs and outputs
    model = Model(inputs=[inp,  bias_profile_input, bias_counts_input],
                  outputs=[profile_out, count_out])
    
    # if only want counts output and input without control tracks
    #ct = Dense(1)(gap_combined_conv)
    #model = Model(inputs=inp, outputs=ct)
    
    return model
