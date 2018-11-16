import argparse
import tensorflow as tf
import cv2
import numpy as np
import model_cc as models_cc
import model_mlo as models_mlo
import utils


def inference(parameters, verbose=True):
    """
    Function that creates a model, loads the parameters, and makes a prediction
    :param parameters: dictionary of parameters
    :param verbose: Whether to print predicted probabilities
    :return: Predicted probabilities for each class
    """

    model_path = 'E:\model_b/agp_birads_model_b.ckpt'
    device_type = "cpu"
    gpu_number = 0
    image_path = "C:/Users/archit.kushwaha.AGILiAD\PycharmProjects\model_d\images/"
    tf.set_random_seed(7)

    with tf.Graph().as_default():
        with tf.device('/' + device_type):
            # initialize input holders

            x_r_cc = tf.placeholder(tf.float32, shape=[1, 2000, 2600, 1])
            x =  x_r_cc
            # holders for dropout and Gaussian noise
            nodropout_probability = tf.placeholder(tf.float32, shape=())
            gaussian_noise_std = tf.placeholder(tf.float32, shape=())

            # construct models
            model_cc = models_cc.BaselineBreastModel(x, nodropout_probability, gaussian_noise_std)
            y_prediction_birads_cc = model_cc.y_prediction_birads

            #model_mlo = models_mlo.BaselineBreastModel(parameters, x, nodropout_probability, gaussian_noise_std)
            #y_prediction_birads_mlo = model_mlo.y_prediction_birads
        if parameters['device_type'] == 'gpu':
            session_config = tf.ConfigProto()
            session_config.gpu_options.visible_device_list = str(parameters['gpu_number'])
        elif parameters['device_type'] == 'cpu':
            session_config = tf.ConfigProto(device_count={'GPU': 0})
        else:
            raise RuntimeError(parameters['device_type'])

        with tf.Session(config=session_config) as session:
            session.run(tf.global_variables_initializer())


        with tf.Session(config=session_config) as session:
            session.run(tf.global_variables_initializer())

            # loads the pre-trained parameters if it's provided
            saver = tf.train.Saver(max_to_keep=None)
            saver.restore(session, 'E:\model_b/agp_birads_model_b.ckpt')

            # load input images
            datum_r_cc = cv2.imread("C:/Users/archit.kushwaha.AGILiAD\PycharmProjects\model_d\images/_00079_RIGHT_CC.png",0)
            datum_r_cc = cv2.resize(datum_r_cc, (2000, 2600))
            datum_r_cc = np.array(datum_r_cc).reshape(1, 2000, 2600, 1)


            # populate feed_dict for TF session
            # No dropout and no gaussian noise in inference
            feed_dict_by_model = {
                nodropout_probability: 1.0,
                gaussian_noise_std: 0.0,
                x_r_cc: datum_r_cc
            }

            # run the session for a prediction
            view_type = "CC"
            if view_type == "CC":
                prediction_birads = session.run(y_prediction_birads_cc, feed_dict=feed_dict_by_model)

            if view_type == "MLO":
                prediction_birads = session.run(y_prediction_birads_cc, feed_dict=feed_dict_by_model)

            if verbose:
                # nicely prints out the predictions
                birads0_prob = prediction_birads[0][0]
                birads1_prob = prediction_birads[0][1]
                birads2_prob = prediction_birads[0][2]
                print('BI-RADS prediction:\n' +
                      '\tBI-RADS 0:\t' + str(birads0_prob) + '\n' +
                      '\tBI-RADS 1:\t' + str(birads1_prob) + '\n' +
                      '\tBI-RADS 2:\t' + str(birads2_prob))

            return prediction_birads[0]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run Inference')
    parser.add_argument('--model-path', default='saved_models/agp_birads.ckpt')
    parser.add_argument('--device-type', default="cpu")
    parser.add_argument('--gpu-number', default=0, type=int)
    parser.add_argument('--image-path', default="images/")
    args = parser.parse_args()

    parameters_ = {
        "model_path": args.model_path,
        "device_type": args.device_type,
        "gpu_number": args.gpu_number,
        "image_path": args.image_path,
        "input_size": (2600, 2000),
    }

    # do a sample prediction
    inference(parameters_)
