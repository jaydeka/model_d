import tensorflow as tf

view_type = "CC"
def all_views_conv_layer(input_layer,view_type , layer_name, number_of_filters=32, filter_size=(3, 3), stride=(1, 1),
                         padding='VALID', biases_initializer=tf.zeros_initializer()):
    """Convolutional layers for 2 views input DCN"""
    view_type = "CC"


    if view_type == "CC":
        output = tf.contrib.layers.convolution2d(inputs=input_layer, num_outputs=number_of_filters,
                                             kernel_size=filter_size, stride=stride, padding=padding, reuse=False,
                                             weights_initializer=tf.contrib.layers.xavier_initializer(),
                                             biases_initializer=biases_initializer)
    if view_type == "MLO":
        output = tf.contrib.layers.convolution2d(inputs=input_layer, num_outputs=number_of_filters,
                                             kernel_size=filter_size, stride=stride, padding=padding, reuse=False,
                                             weights_initializer=tf.contrib.layers.xavier_initializer(),
                                             biases_initializer=biases_initializer)

    return output


def all_views_max_pool(input_layer, view_type, stride=(2, 2)):
    """Max-pool across all views"""
    view_type = "CC"
    if view_type == "CC":
        output = tf.nn.max_pool(input_layer, ksize=[1, stride[0], stride[1], 1], strides=[1, stride[0], stride[1], 1],
                                 padding='SAME')
    if view_type == "MLO":
        output = tf.nn.max_pool(input_layer, ksize=[1, stride[0], stride[1], 1], strides=[1, stride[0], stride[1], 1],
                                     padding='SAME')
    return output


def all_views_global_avg_pool(input_layer, view_type):
    """Average-pool across all views"""
    view_type = "CC"

    if view_type == "CC":
        input_layer_shape = input_layer.get_shape()
        pooling_shape = [1, input_layer_shape[1], input_layer_shape[2], 1]
        output = tf.nn.avg_pool(input_layer, ksize=pooling_shape, strides=pooling_shape, padding='SAME')

    if view_type == "MLO":
        input_layer_shape = input_layer.get_shape()
        pooling_shape = [1, input_layer_shape[1], input_layer_shape[2], 1]
        output = tf.nn.avg_pool(input_layer, ksize=pooling_shape, strides=pooling_shape, padding='SAME')

    return output


def all_views_flattening_layer(input_layer,view_type):

    """Flattenall activations from all  views"""
    view_type = "CC"

    if view_type == "CC":
        input_layer_shape = input_layer.get_shape()
        input_layer_size = int(input_layer_shape[1]) * int(input_layer_shape[2]) * int(input_layer_shape[3])
        output_flat = tf.reshape(input_layer, [-1, input_layer_size])

    if view_type == "MLO":
        input_layer_shape = input_layer.get_shape()
        input_layer_size = int(input_layer_shape[1]) * int(input_layer_shape[2]) * int(input_layer_shape[3])
        output_flat = tf.reshape(input_layer, [-1, input_layer_size])

    return output_flat


def fc_layer(input_layer, number_of_units=128, activation_fn=tf.nn.relu, reuse=None, scope=None):
    """Fully connected layer"""

    h = tf.contrib.layers.fully_connected(inputs=input_layer, num_outputs=number_of_units, activation_fn=activation_fn,
                                              reuse=reuse, scope=scope)
    return h


def softmax_layer(input_layer, number_of_outputs=3):
    """Softmax layer"""

    y_prediction = tf.contrib.layers.fully_connected(inputs=input_layer, num_outputs=number_of_outputs,
                                                            activation_fn=tf.nn.softmax)
    return y_prediction


def dropout_layer(input_layer, nodropout_probability=0.50):
    """Dropout layer"""
    output = tf.nn.dropout(input_layer, nodropout_probability)

    return output


def gaussian_noise_layer(input_layer, std):
    """Additive gaussian noise layer"""

    noise = tf.random_normal(tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)

    output = tf.add_n([input_layer, noise])

    return output


def all_views_gaussian_noise_layer(input_layer, std):
    """Add gaussian noise across all 4 views"""

    output = gaussian_noise_layer(input_layer, std)

    return output
