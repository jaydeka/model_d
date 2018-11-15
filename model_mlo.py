
#import model_b.layers as layers
import layers_mlo as layers


def baseline(x,view_type,   parameters, nodropout_probability=None, gaussian_noise_std=None):
    #x,view_type = input

    if gaussian_noise_std is not None:
        x = layers.all_views_gaussian_noise_layer(x, gaussian_noise_std)

    # first conv sequence
    h = layers.all_views_conv_layer(x, view_type, 'conv1', number_of_filters=32, filter_size=[3, 3], stride=[2, 2])

    # second conv sequence
    h = layers.all_views_max_pool(h, view_type,  stride=[3, 3])
    h = layers.all_views_conv_layer(h, view_type,  'conv2a', number_of_filters=64, filter_size=[3, 3], stride=[2, 2])
    h = layers.all_views_conv_layer(h, view_type,  'conv2b', number_of_filters=64, filter_size=[3, 3], stride=[1, 1])
    h = layers.all_views_conv_layer(h, view_type,  'conv2c', number_of_filters=64, filter_size=[3, 3], stride=[1, 1])

    # third conv sequence
    h = layers.all_views_max_pool(h, view_type, stride=[2, 2])
    h = layers.all_views_conv_layer(h, view_type,  'conv3a', number_of_filters=128, filter_size=[3, 3], stride=[1, 1])
    h = layers.all_views_conv_layer(h, view_type, 'conv3b', number_of_filters=128, filter_size=[3, 3], stride=[1, 1])
    h = layers.all_views_conv_layer(h, view_type,  'conv3c', number_of_filters=128, filter_size=[3, 3], stride=[1, 1])

    # fourth conv sequence
    h = layers.all_views_max_pool(h, view_type,  stride=[2, 2])
    h = layers.all_views_conv_layer(h, view_type,  'conv4a', number_of_filters=128, filter_size=[3, 3], stride=[1, 1])
    h = layers.all_views_conv_layer(h, view_type, 'conv4b', number_of_filters=128, filter_size=[3, 3], stride=[1, 1])
    h = layers.all_views_conv_layer(h, view_type, 'conv4c', number_of_filters=128, filter_size=[3, 3], stride=[1, 1])

    # fifth conv sequence
    h = layers.all_views_max_pool(h, view_type, stride=[2, 2])
    h = layers.all_views_conv_layer(h, view_type, 'conv5a', number_of_filters=256, filter_size=[3, 3], stride=[1, 1])
    h = layers.all_views_conv_layer(h, view_type, 'conv5b', number_of_filters=256, filter_size=[3, 3], stride=[1, 1])
    h = layers.all_views_conv_layer(h, view_type, 'conv5c', number_of_filters=256, filter_size=[3, 3], stride=[1, 1])

    # Pool, flatten, and fully connected layers
    h = layers.all_views_global_avg_pool(h, view_type)
    h = layers.all_views_flattening_layer(h, view_type) #flatening and concatenation
    h = layers.fc_layer(h, number_of_units=1024)

    #h = layers.dropout_layer(h, nodropout_probability)

    y_prediction_birads = layers.softmax_layer(h, number_of_outputs=3)
    print(y_prediction_birads)

    return y_prediction_birads


class BaselineBreastModel:

    def __init__(self, x, view_type,parameters, nodropout_probability=0.5, gaussian_noise_std=1):
        self.y_prediction_birads = baseline(x, view_type, parameters, nodropout_probability, gaussian_noise_std)