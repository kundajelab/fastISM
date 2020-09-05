import tensorflow as tf


def bpnet_model(seqlen=1000, numchars=4, num_dilated_convs=9, num_tasks=1,
                name='bpnet_model'):
    # original as per https://www.biorxiv.org/content/10.1101/737981v1.full.pdf
    inp = tf.keras.layers.Input(shape=(seqlen, 4))
    x = tf.keras.layers.Conv1D(
        64, kernel_size=25, padding='same', activation='relu')(inp)
    
    for i in range(num_dilated_convs):
        conv_x = tf.keras.layers.Conv1D(
            64, kernel_size=3, padding='same', activation='relu', dilation_rate=2**i)(x)
        x = tf.keras.layers.Add()([conv_x, x])
    bottleneck = x
    
    # heads
    outputs = []
    for _ in range(num_tasks):
        # profile shape head
        px = tf.keras.layers.Reshape((-1, 1, 64))(bottleneck)
        px = tf.keras.layers.Conv2DTranspose(
            1, kernel_size=(25, 1), padding='same')(px)
        outputs.append(tf.keras.layers.Flatten()(px))

        # total counts head
        cx = tf.keras.layers.GlobalAvgPool1D()(bottleneck)
        outputs.append(tf.keras.layers.Dense(1)(cx))

    model = tf.keras.Model(inputs=inp, outputs=outputs)

    return model
