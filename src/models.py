import tensorflow as tf
from keras.layers import concatenate
from keras.models import Model
from keras.regularizers import l2
from keras_cv_attention_models.attention_layers import (
    MultiHeadRelativePositionalEmbedding,
    batchnorm_with_activation,
    conv2d_no_bias,
    depthwise_conv2d_no_bias,
    drop_block,
    se_module,
)
from tensorflow import keras


def ResNet152V2():
    model = tf.keras.applications.ResNet152V2(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=2,
        classifier_activation="softmax",
    )
    return model


def InceptionResNetV2():
    model = tf.keras.applications.InceptionResNetV2(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=2,
        classifier_activation="softmax",
    )
    return model


def InceptionV3():
    model = tf.keras.applications.InceptionV3(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=2,
        classifier_activation="softmax",
    )
    return model


def MobileNetV2():
    model = tf.keras.applications.MobileNetV2(
        input_shape=(424, 424, 3),
        alpha=1.0,
        include_top=True,
        weights=None,
        input_tensor=None,
        pooling=None,
        classes=2,
        classifier_activation="softmax",
        # **kwargs
    )
    return model


def Alexnet():
    model = keras.models.Sequential(
        [
            keras.layers.Conv2D(
                filters=96,
                kernel_size=(11, 11),
                strides=(4, 4),
                activation="relu",
                input_shape=(227, 227, 3),
            ),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            keras.layers.Conv2D(
                filters=256,
                kernel_size=(5, 5),
                strides=(1, 1),
                activation="relu",
                padding="same",
            ),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            keras.layers.Conv2D(
                filters=384,
                kernel_size=(3, 3),
                strides=(1, 1),
                activation="relu",
                padding="same",
            ),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(
                filters=384,
                kernel_size=(3, 3),
                strides=(1, 1),
                activation="relu",
                padding="same",
            ),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(
                filters=256,
                kernel_size=(3, 3),
                strides=(1, 1),
                activation="relu",
                padding="same",
            ),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(4096, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(4096, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(
                2, activation="softmax"
            ),  # binário: sigmoid = soma não da 1 | multclass: softmax = soma: 1
        ]
    )
    return model


def residual_block(
    input,
    input_channels=None,
    output_channels=None,
    kernel_size=(3, 3),
    stride=1,
):
    """
    full pre-activation residual block
    https://arxiv.org/pdf/1603.05027.pdf
    """
    if output_channels is None:
        output_channels = input.get_shape().as_list()[-1]
    if input_channels is None:
        input_channels = output_channels // 4

    strides = (stride, stride)

    x = keras.layers.BatchNormalization()(input)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(input_channels, (1, 1))(x)

    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(
        input_channels, kernel_size, padding="same", strides=stride
    )(x)

    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(output_channels, (1, 1), padding="same")(x)

    if input_channels != output_channels or stride != 1:
        input = keras.layers.Conv2D(
            output_channels, (1, 1), padding="same", strides=strides
        )(input)

    x = keras.layers.Add()([x, input])
    return x


def attention_block(input, input_channels=None, output_channels=None, encoder_depth=1):
    """
    attention block
    https://arxiv.org/abs/1704.06904
    """

    p = 1
    t = 2
    r = 1

    if input_channels is None:
        input_channels = input.get_shape().as_list()[-1]
    if output_channels is None:
        output_channels = input_channels

    # First Residual Block
    for i in range(p):
        input = residual_block(input)

    # Trunc Branch
    output_trunk = input
    for i in range(t):
        output_trunk = residual_block(output_trunk)

    # Soft Mask Branch

    # encoder
    # first down sampling
    output_soft_mask = keras.layers.MaxPool2D(padding="same")(input)  # 32x32
    for i in range(r):
        output_soft_mask = residual_block(output_soft_mask)

    skip_connections = []
    for i in range(encoder_depth - 1):
        # skip connections
        output_skip_connection = residual_block(output_soft_mask)
        skip_connections.append(output_skip_connection)
        # print ('skip shape:', output_skip_connection.get_shape())

        # down sampling
        output_soft_mask = keras.layers.MaxPool2D(padding="same")(output_soft_mask)
        for _ in range(r):
            output_soft_mask = residual_block(output_soft_mask)

            # decoder
    skip_connections = list(reversed(skip_connections))
    for i in range(encoder_depth - 1):
        # upsampling
        for _ in range(r):
            output_soft_mask = residual_block(output_soft_mask)
        output_soft_mask = keras.layers.UpSampling2D()(output_soft_mask)
        # skip connections
        output_soft_mask = keras.layers.Add()([output_soft_mask, skip_connections[i]])

    # last upsampling
    for i in range(r):
        output_soft_mask = residual_block(output_soft_mask)
    output_soft_mask = keras.layers.UpSampling2D()(output_soft_mask)

    # Output
    output_soft_mask = keras.layers.Conv2D(input_channels, (1, 1))(output_soft_mask)
    output_soft_mask = keras.layers.Conv2D(input_channels, (1, 1))(output_soft_mask)
    output_soft_mask = keras.layers.Activation("softmax")(
        output_soft_mask
    )  # binário = sigmoid | multclass = softmax

    # Attention: (1 + output_soft_mask) * output_trunk
    output = keras.layers.Lambda(lambda x: x + 1)(output_soft_mask)
    output = keras.layers.Multiply()([output, output_trunk])

    # Last Residual Block
    for i in range(p):
        output = residual_block(output)

    return output


def AttentionResNet56(
    shape=(224, 224, 3),
    n_channels=64,
    n_classes=2,
    dropout=0,
    regularization=0.01,
):
    """
    Attention-56 ResNet
    https://arxiv.org/abs/1704.06904
    """

    regularizer = l2(regularization)

    input_ = keras.layers.Input(shape=shape)
    x = keras.layers.Conv2D(n_channels, (7, 7), strides=(2, 2), padding="same")(
        input_
    )  # 112x112
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same")(
        x
    )  # 56x56

    x = residual_block(x, output_channels=n_channels * 4)  # 56x56
    x = attention_block(x, encoder_depth=3)  # bottleneck 7x7

    x = residual_block(x, output_channels=n_channels * 8, stride=2)  # 28x28
    x = attention_block(x, encoder_depth=2)  # bottleneck 7x7

    x = residual_block(x, output_channels=n_channels * 16, stride=2)  # 14x14
    x = attention_block(x, encoder_depth=1)  # bottleneck 7x7

    x = residual_block(x, output_channels=n_channels * 32, stride=2)  # 7x7
    x = residual_block(x, output_channels=n_channels * 32)
    x = residual_block(x, output_channels=n_channels * 32)

    pool_size = (x.get_shape().as_list()[1], x.get_shape().as_list()[2])
    x = keras.layers.AveragePooling2D(pool_size=pool_size, strides=(1, 1))(x)
    x = keras.layers.Flatten()(x)
    if dropout:
        x = keras.layers.Dropout(dropout)(x)
    output = keras.layers.Dense(
        n_classes, kernel_regularizer=regularizer, activation="softmax"
    )(x)

    model = Model(input_, output)
    return model


def AttentionResNet92(
    shape=(224, 224, 3),
    n_channels=64,
    n_classes=2,
    dropout=0,
    regularization=0.01,
):
    """
    Attention-92 ResNet
    https://arxiv.org/abs/1704.06904
    """
    regularizer = l2(regularization)

    input_ = keras.layers.Input(shape=shape)
    x = keras.layers.Conv2D(n_channels, (7, 7), strides=(2, 2), padding="same")(
        input_
    )  # 112x112
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same")(
        x
    )  # 56x56

    x = residual_block(x, output_channels=n_channels * 4)  # 56x56
    x = attention_block(x, encoder_depth=3)  # bottleneck 7x7

    x = residual_block(x, output_channels=n_channels * 8, stride=2)  # 28x28
    x = attention_block(x, encoder_depth=2)  # bottleneck 7x7
    x = attention_block(x, encoder_depth=2)  # bottleneck 7x7

    x = residual_block(x, output_channels=n_channels * 16, stride=2)  # 14x14
    x = attention_block(x, encoder_depth=1)  # bottleneck 7x7
    x = attention_block(x, encoder_depth=1)  # bottleneck 7x7
    x = attention_block(x, encoder_depth=1)  # bottleneck 7x7

    x = residual_block(x, output_channels=n_channels * 32, stride=2)  # 7x7
    x = residual_block(x, output_channels=n_channels * 32)
    x = residual_block(x, output_channels=n_channels * 32)

    pool_size = (x.get_shape().as_list()[1], x.get_shape().as_list()[2])
    x = keras.layers.AveragePooling2D(pool_size=pool_size, strides=(1, 1))(x)
    x = keras.layers.Flatten()(x)
    if dropout:
        x = keras.layers.Dropout(dropout)(x)
    output = keras.layers.Dense(
        n_classes, kernel_regularizer=regularizer, activation="softmax"
    )(x)
    model = Model(input_, output)
    return model


def Inception_block(input_layer, f1, f2_conv1, f2_conv3, f3_conv1, f3_conv5, f4):
    # Input:
    # - f1: number of filters of the 1x1 convolutional layer in the first path
    # - f2_conv1, f2_conv3 are number of filters corresponding to the 1x1 and 3x3 convolutional layers in the second path
    # - f3_conv1, f3_conv5 are the number of filters corresponding to the 1x1 and 5x5  convolutional layer in the third path
    # - f4: number of filters of the 1x1 convolutional layer in the fourth path

    # 1st path:
    path1 = keras.layers.Conv2D(
        filters=f1, kernel_size=(1, 1), padding="same", activation="relu"
    )(input_layer)

    # 2nd path
    path2 = keras.layers.Conv2D(
        filters=f2_conv1, kernel_size=(1, 1), padding="same", activation="relu"
    )(input_layer)
    path2 = keras.layers.Conv2D(
        filters=f2_conv3, kernel_size=(3, 3), padding="same", activation="relu"
    )(path2)

    # 3rd path
    path3 = keras.layers.Conv2D(
        filters=f3_conv1, kernel_size=(1, 1), padding="same", activation="relu"
    )(input_layer)
    path3 = keras.layers.Conv2D(
        filters=f3_conv5, kernel_size=(5, 5), padding="same", activation="relu"
    )(path3)

    # 4th path
    path4 = keras.layers.MaxPooling2D((3, 3), strides=(1, 1), padding="same")(
        input_layer
    )
    path4 = keras.layers.Conv2D(
        filters=f4, kernel_size=(1, 1), padding="same", activation="relu"
    )(path4)

    output_layer = concatenate([path1, path2, path3, path4], axis=-1)

    return output_layer


def GoogLeNet():
    # input layer
    input_layer = keras.layers.Input(shape=(224, 224, 3))

    # convolutional layer: filters = 64, kernel_size = (7,7), strides = 2
    X = keras.layers.Conv2D(
        filters=64,
        kernel_size=(7, 7),
        strides=2,
        padding="valid",
        activation="relu",
    )(input_layer)

    # max-pooling layer: pool_size = (3,3), strides = 2
    X = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2)(X)

    # convolutional layer: filters = 64, strides = 1
    X = keras.layers.Conv2D(
        filters=64,
        kernel_size=(1, 1),
        strides=1,
        padding="same",
        activation="relu",
    )(X)

    # convolutional layer: filters = 192, kernel_size = (3,3)
    X = keras.layers.Conv2D(
        filters=192, kernel_size=(3, 3), padding="same", activation="relu"
    )(X)

    # max-pooling layer: pool_size = (3,3), strides = 2
    X = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2)(X)

    # 1st Inception block
    X = Inception_block(
        X, f1=64, f2_conv1=96, f2_conv3=128, f3_conv1=16, f3_conv5=32, f4=32
    )

    # 2nd Inception block
    X = Inception_block(
        X, f1=128, f2_conv1=128, f2_conv3=192, f3_conv1=32, f3_conv5=96, f4=64
    )

    # max-pooling layer: pool_size = (3,3), strides = 2
    X = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2)(X)

    # 3rd Inception block
    X = Inception_block(
        X, f1=192, f2_conv1=96, f2_conv3=208, f3_conv1=16, f3_conv5=48, f4=64
    )

    # Extra network 1:
    X1 = keras.layers.AveragePooling2D(pool_size=(5, 5), strides=3)(X)
    X1 = keras.layers.Conv2D(
        filters=128, kernel_size=(1, 1), padding="same", activation="relu"
    )(X1)
    X1 = keras.layers.Flatten()(X1)
    X1 = keras.layers.Dense(1024, activation="relu")(X1)
    X1 = keras.layers.Dropout(0.7)(X1)
    X1 = keras.layers.Dense(2, activation="softmax")(X1)

    # 4th Inception block
    X = Inception_block(
        X, f1=160, f2_conv1=112, f2_conv3=224, f3_conv1=24, f3_conv5=64, f4=64
    )

    # 5th Inception block
    X = Inception_block(
        X, f1=128, f2_conv1=128, f2_conv3=256, f3_conv1=24, f3_conv5=64, f4=64
    )

    # 6th Inception block
    X = Inception_block(
        X, f1=112, f2_conv1=144, f2_conv3=288, f3_conv1=32, f3_conv5=64, f4=64
    )

    # Extra network 2:
    X2 = keras.layers.AveragePooling2D(pool_size=(5, 5), strides=3)(X)
    X2 = keras.layers.Conv2D(
        filters=128, kernel_size=(1, 1), padding="same", activation="relu"
    )(X2)
    X2 = keras.layers.Flatten()(X2)
    X2 = keras.layers.Dense(1024, activation="relu")(X2)
    X2 = keras.layers.Dropout(0.7)(X2)
    X2 = keras.layers.Dense(2, activation="softmax")(X2)

    # 7th Inception block
    X = Inception_block(
        X,
        f1=256,
        f2_conv1=160,
        f2_conv3=320,
        f3_conv1=32,
        f3_conv5=128,
        f4=128,
    )

    # max-pooling layer: pool_size = (3,3), strides = 2
    X = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2)(X)

    # 8th Inception block
    X = Inception_block(
        X,
        f1=256,
        f2_conv1=160,
        f2_conv3=320,
        f3_conv1=32,
        f3_conv5=128,
        f4=128,
    )

    # 9th Inception block
    X = Inception_block(
        X,
        f1=384,
        f2_conv1=192,
        f2_conv3=384,
        f3_conv1=48,
        f3_conv5=128,
        f4=128,
    )

    # Global Average pooling layer
    X = keras.layers.GlobalAveragePooling2D(name="GAPL")(X)

    # Dropoutlayer
    X = keras.layers.Dropout(0.4)(X)

    # output layer
    X = keras.layers.Dense(2, activation="softmax")(X)

    # model
    # model = Model(input_layer, [X, X1, X2], name = 'GoogLeNet')

    model = Model(input_layer, [X, X1, X2], name="GoogLeNet")

    return model


def mhsa_with_multi_head_relative_position_embedding(
    inputs,
    num_heads=4,
    key_dim=0,
    relative=True,
    out_shape=None,
    out_weight=True,
    out_bias=False,
    attn_dropout=0,
    name=None,
):
    _, hh, ww, cc = inputs.shape
    key_dim = key_dim if key_dim > 0 else cc // num_heads
    qk_scale = 1.0 / tf.math.sqrt(tf.cast(key_dim, inputs.dtype))
    out_shape = cc if out_shape is None or not out_weight else out_shape
    qk_out = num_heads * key_dim
    vv_dim = out_shape // num_heads
    # final_out_shape = (None, hh, ww, out_shape)

    # qkv = keras.layers.Dense(emb_dim * 3, use_bias=False, name=name and name + "qkv")(inputs)
    qkv = conv2d_no_bias(
        inputs,
        qk_out * 2 + out_shape,
        kernel_size=1,
        name=name and name + "qkv_",
    )
    qkv = tf.reshape(qkv, [-1, inputs.shape[1] * inputs.shape[2], qkv.shape[-1]])
    query, key, value = tf.split(qkv, [qk_out, qk_out, out_shape], axis=-1)
    # query = [batch, num_heads, hh * ww, key_dim]
    query = tf.transpose(
        tf.reshape(query, [-1, query.shape[1], num_heads, key_dim]),
        [0, 2, 1, 3],
    )
    # key = [batch, num_heads, key_dim, hh * ww]
    key = tf.transpose(
        tf.reshape(key, [-1, key.shape[1], num_heads, key_dim]), [0, 2, 3, 1]
    )
    # value = [batch, num_heads, hh * ww, vv_dim]
    value = tf.transpose(
        tf.reshape(value, [-1, value.shape[1], num_heads, vv_dim]),
        [0, 2, 1, 3],
    )

    # query *= qk_scale
    # [batch, num_heads, hh * ww, hh * ww]
    attention_scores = (
        keras.layers.Lambda(lambda xx: tf.matmul(xx[0], xx[1]))([query, key]) * qk_scale
    )
    attention_scores = MultiHeadRelativePositionalEmbedding(
        with_cls_token=False, name=name and name + "pos_emb"
    )(attention_scores)
    attention_scores = tf.nn.softmax(attention_scores, axis=-1)

    if attn_dropout > 0:
        attention_scores = keras.layers.Dropout(
            attn_dropout, name=name and name + "attn_drop"
        )(attention_scores)
    # value = [batch, num_heads, hh * ww, vv_dim]
    # attention_output = tf.matmul(attention_scores, value)  # [batch, num_heads, hh * ww, vv_dim]
    attention_output = keras.layers.Lambda(lambda xx: tf.matmul(xx[0], xx[1]))(
        [attention_scores, value]
    )
    attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
    attention_output = tf.reshape(
        attention_output,
        [-1, inputs.shape[1], inputs.shape[2], num_heads * vv_dim],
    )
    # print(f">>>> {attention_output.shape = }, {attention_scores.shape = }")

    if out_weight:
        # [batch, hh, ww, num_heads * vv_dim] * [num_heads * vv_dim, out] --> [batch, hh, ww, out]
        attention_output = keras.layers.Dense(
            out_shape, use_bias=out_bias, name=name and name + "output"
        )(attention_output)
    # attention_output.set_shape(final_out_shape)
    return attention_output


def res_MBConv(
    inputs,
    output_channel,
    conv_short_cut=True,
    strides=1,
    expansion=4,
    se_ratio=0,
    drop_rate=0,
    activation="gelu",
    name="",
):
    """x ← Proj(Pool(x)) + Conv (DepthConv (Conv (Norm(x), stride = 2))))"""
    # preact
    preact = batchnorm_with_activation(
        inputs, activation=activation, zero_gamma=False, name=name + "preact_"
    )

    if conv_short_cut:
        # Avg or Max pool
        # shortcut = keras.layers.AvgPool2D(strides, strides=strides, padding="SAME", name=name + "shortcut_pool")(inputs) if strides > 1 else inputs
        shortcut = (
            keras.layers.AvgPool2D(
                strides,
                strides=strides,
                padding="SAME",
                name=name + "shortcut_pool",
            )(preact)
            if strides > 1
            else preact
        )
        shortcut = conv2d_no_bias(
            shortcut, output_channel, 1, strides=1, name=name + "shortcut_"
        )
    else:
        shortcut = inputs

    # MBConv
    input_channel = inputs.shape[-1]
    nn = conv2d_no_bias(
        preact,
        input_channel * expansion,
        1,
        strides=1,
        padding="same",
        name=name + "expand_",
    )  # May swap stirdes with DW
    nn = batchnorm_with_activation(nn, activation=activation, name=name + "expand_")
    nn = depthwise_conv2d_no_bias(
        nn, 3, strides=strides, padding="same", name=name + "MB_"
    )
    nn = batchnorm_with_activation(nn, activation=activation, name=name + "MB_dw_")
    if se_ratio:
        nn = se_module(
            nn,
            se_ratio=se_ratio / expansion,
            activation=activation,
            name=name + "se_",
        )
    nn = conv2d_no_bias(
        nn, output_channel, 1, strides=1, padding="same", name=name + "MB_pw_"
    )
    nn = drop_block(nn, drop_rate=drop_rate, name=name)
    return keras.layers.Add()([shortcut, nn])


def res_ffn(
    inputs, expansion=4, kernel_size=1, drop_rate=0, activation="gelu", name=""
):
    """x ← x + Module (Norm(x))"""
    # preact
    preact = batchnorm_with_activation(
        inputs, activation=activation, zero_gamma=False, name=name + "preact_"
    )
    # nn = layer_norm(inputs, name=name + "preact_")

    input_channel = inputs.shape[-1]
    nn = conv2d_no_bias(
        preact, input_channel * expansion, kernel_size, name=name + "1_"
    )
    # nn = activation_by_name(nn, activation=activation, name=name)
    nn = batchnorm_with_activation(nn, activation=activation, name=name + "ffn_")
    nn = conv2d_no_bias(nn, input_channel, kernel_size, name=name + "2_")
    nn = drop_block(nn, drop_rate=drop_rate, name=name)
    return keras.layers.Add()([inputs, nn])


def res_mhsa(
    inputs,
    output_channel,
    conv_short_cut=True,
    strides=1,
    head_dimension=32,
    drop_rate=0,
    activation="gelu",
    name="",
):
    """x ← Proj(Pool(x)) + Attention (Pool(Norm(x)))"""
    # preact
    preact = batchnorm_with_activation(
        inputs, activation=activation, zero_gamma=False, name=name + "preact_"
    )
    # preact = layer_norm(inputs, name=name + "preact_")

    if conv_short_cut:
        # Avg or Max pool
        # shortcut = keras.layers.AvgPool2D(strides, strides=strides, padding="SAME", name=name + "shortcut_pool")(inputs) if strides > 1 else inputs
        shortcut = (
            keras.layers.AvgPool2D(
                strides,
                strides=strides,
                padding="SAME",
                name=name + "shortcut_pool",
            )(preact)
            if strides > 1
            else preact
        )
        shortcut = conv2d_no_bias(
            shortcut, output_channel, 1, strides=1, name=name + "shortcut_"
        )
    else:
        shortcut = inputs

    nn = preact
    if strides != 1:  # Downsample
        # nn = keras.layers.ZeroPadding2D(padding=1, name=name + "pad")(nn)
        nn = keras.layers.MaxPool2D(
            pool_size=2, strides=strides, padding="SAME", name=name + "pool"
        )(nn)
    num_heads = nn.shape[-1] // head_dimension
    nn = mhsa_with_multi_head_relative_position_embedding(
        nn,
        num_heads=num_heads,
        key_dim=head_dimension,
        out_shape=output_channel,
        name=name + "mhsa",
    )
    nn = drop_block(nn, drop_rate=drop_rate, name=name)
    # print(f"{name = }, {inputs.shape = }, {shortcut.shape = }, {nn.shape = }")
    return keras.layers.Add()([shortcut, nn])


def CoAtNet(
    num_blocks,
    out_channels,
    stem_width=64,
    block_types=["conv", "conv", "transfrom", "transform"],
    strides=[2, 2, 2, 2],
    expansion=4,
    se_ratio=0.25,
    head_dimension=32,
    input_shape=(224, 224, 3),
    num_classes=2,
    activation="gelu",
    drop_connect_rate=0,
    classifier_activation="softmax",
    drop_rate=0,
    pretrained=None,
    model_name="coatnet",
    kwargs=None,
):
    inputs = keras.layers.Input(input_shape)

    """ stage 0, Stem_stage """
    nn = conv2d_no_bias(
        inputs, stem_width, 3, strides=2, padding="same", name="stem_1_"
    )
    nn = batchnorm_with_activation(nn, activation=activation, name="stem_1_")
    nn = conv2d_no_bias(nn, stem_width, 3, strides=1, padding="same", name="stem_2_")

    """ stage [1, 2, 3, 4] """
    total_blocks = sum(num_blocks)
    global_block_id = 0
    for stack_id, (num_block, out_channel, block_type) in enumerate(
        zip(num_blocks, out_channels, block_types)
    ):
        is_conv_block = True if block_type[0].lower() == "c" else False
        stack_se_ratio = (
            se_ratio[stack_id] if isinstance(se_ratio, (list, tuple)) else se_ratio
        )
        stack_strides = (
            strides[stack_id] if isinstance(strides, (list, tuple)) else strides
        )
        for block_id in range(num_block):
            name = "stage_{}_block_{}_".format(stack_id + 1, block_id + 1)
            stride = stack_strides if block_id == 0 else 1
            conv_short_cut = True if block_id == 0 else False
            block_se_ratio = (
                stack_se_ratio[block_id]
                if isinstance(stack_se_ratio, (list, tuple))
                else stack_se_ratio
            )
            block_drop_rate = drop_connect_rate * global_block_id / total_blocks
            global_block_id += 1
            if is_conv_block:
                nn = res_MBConv(
                    nn,
                    out_channel,
                    conv_short_cut,
                    stride,
                    expansion,
                    block_se_ratio,
                    block_drop_rate,
                    activation=activation,
                    name=name,
                )
            else:
                nn = res_mhsa(
                    nn,
                    out_channel,
                    conv_short_cut,
                    stride,
                    head_dimension,
                    block_drop_rate,
                    activation=activation,
                    name=name,
                )
                nn = res_ffn(
                    nn,
                    expansion=expansion,
                    drop_rate=block_drop_rate,
                    activation=activation,
                    name=name + "ffn_",
                )

    if num_classes > 0:
        nn = keras.layers.GlobalAveragePooling2D(name="avg_pool")(nn)
        if drop_rate > 0:
            nn = keras.layers.Dropout(drop_rate)(nn)
        nn = keras.layers.Dense(
            num_classes,
            dtype="float32",
            activation=classifier_activation,
            name="predictions",
        )(nn)

    model = keras.models.Model(inputs, nn, name=model_name)
    return model


def CoAtNet4(
    input_shape=(224, 224, 3),
    num_classes=2,
    activation="gelu",
    classifier_activation="softmax",
    **kwargs,
):
    num_blocks = [2, 12, 28, 2]
    out_channels = [192, 384, 768, 1536]
    stem_width = 192
    return CoAtNet(**locals(), model_name="coatnet4", **kwargs)
