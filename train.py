import load_data as load_data
import config as config
import tensorflow as tf
import numpy as np
import model_cc as model_cc
import model_mlo as model_mlo


train_dataset_folder = config.folder_path["train_dataset"]
train_xlsx = config.file_path["train_xlsx"]
test_dataset_folder = config.folder_path["test_dataset"]
test_xlsx = config.file_path["test_xlsx"]
training_iters = config.hyperparameter_train["training_iters"]
learning_rate = config.hyperparameter_train["learning_rate"]
batch_size = config.hyperparameter_train["batch_size"]
no_epochs = config.hyperparameter_train["no_epochs"]
n_classes = config.hyperparameter_train["n_classes"]
model_save = config.model_ckpts["saved_model"]

#load train and test dataset
print("Loading train data... ")
array_images, array_target = load_data.load_dataset(train_dataset_folder,train_xlsx)
print("Loading test data... ")
array_images_test, array_target_test =  load_data.load_dataset(test_dataset_folder, test_xlsx)
print("Size of Dataset: ")
print("Training Set : ", len(array_target))
print("Validation Set : ", len(array_images_test) // 2)
print("Test Set : ", len(array_images_test) // 2)

#define placeholders
x_img = tf.placeholder(tf.float32, shape=[1, 2000, 2600, 1])
x_str = tf.placeholder(tf.string, shape=None)
x = (x_img)
y = tf.placeholder(tf.float32, shape=(1, 3))

#train the network
prediction = model_cc.baseline(x, x_str)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
# optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.99, epsilon=0.1)
optimizer2 = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)
train2 = optimizer2.minimize(cost)


prediction_mlo = model_mlo.baseline(x, x_str,  parameters=None)
cost_mlo = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction_mlo, labels=y))
# optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
correct_prediction_mlo = tf.equal(tf.argmax(prediction_mlo, 1), tf.argmax(y, 1))
accuracy_mlo = tf.reduce_mean(tf.cast(correct_prediction_mlo, tf.float32))
optimizer_mlo = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.99, epsilon=0.1)
optimizer2_mlo = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_mlo = optimizer.minimize(cost_mlo)
train2_mlo = optimizer2.minimize(cost_mlo)

init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=1)

with tf.Session() as sess:
    init.run()
    # sess.run(init)
    train_loss = []
    test_loss = []
    valid_loss = []
    valid_accuracy = []
    train_accuracy = []
    test_accuracy = []
    summary_writer = tf.summary.FileWriter(config.model_ckpts["model_output"], sess.graph)
    iterator = 0
    print("Started Training...")

    for epoch in range(no_epochs):
        for j in range(50):
        #for j in range(2):
            train_x1_rcc = array_images[j]
            x1, x2 = train_x1_rcc
            train_y = array_target_test[j]
            if str(x2) == "CC":
                feed_dict_model = {x_img: x1, x_str: x2, y: train_y}
                sess.run(train, feed_dict=feed_dict_model)
                loss, acc = sess.run([cost, accuracy],
                                     feed_dict={x_img: x1, x_str: x2, y: train_y})
                train_accuracy.append(acc)
                train_loss.append(loss)
            if str(x2) == "MLO":
                feed_dict_model = {x_img: x1, x_str: x2, y: train_y}
                sess.run(train, feed_dict=feed_dict_model)
                loss, acc = sess.run([cost_mlo, accuracy_mlo],
                                     feed_dict={x_img: x1, x_str: x2, y: train_y})
                train_accuracy.append(acc)
                train_loss.append(loss)

        for k in range((len(array_images_test) // 2) - 1):
        #for k in range(2):
            test_X = array_images_test[k]
            x1, x2 = test_X
            valid_y = array_target_test[k]
            if str(x2) == "CC":
                feed_dict_model = {x_img:x1, x_str:x2, y: valid_y}
                validation_acc, validation_loss = sess.run([accuracy, cost], feed_dict=feed_dict_model)
                valid_loss.append(validation_loss)
                valid_accuracy.append(validation_acc)
            if str(x2) == "MLO":
                feed_dict_model = {x_img: x1, x_str: x2, y: valid_y}
                validation_acc, validation_loss = sess.run([accuracy_mlo, cost_mlo], feed_dict=feed_dict_model)
                valid_loss.append(validation_loss)
                valid_accuracy.append(validation_acc)

        overall_valid_accuracy = np.mean(valid_accuracy)
        overall_train_accuracy =  np.mean(train_accuracy)
        overall_valid_loss = np.mean(valid_loss)
        overall_train_loss = np.mean(train_loss)
        print("Epoch ", epoch, " Training Accuracy: ", overall_train_accuracy, " Training Loss: ", overall_train_loss, " Validation Accuracy: ", overall_valid_accuracy, " Validation Loss: ", overall_valid_loss)

        for l in range((len(array_images_test) // 2), len(array_images_test), 1):
        #for l in range(2):
            test_X = array_images_test[l]
            x1, x2 = test_X
            test_y = array_target_test[l]
            if str(x2) == "CC":
                feed_dict_model = {x_img:x1, x_str:x2,  y: test_y}
                test_acc_temp, test_loss_temp = sess.run([accuracy, cost], feed_dict=feed_dict_model)
                test_loss.append(test_loss_temp)
                test_accuracy.append(test_acc_temp)
            if str(x2) == "MLO":
                feed_dict_model = {x_img: x1, x_str: x2, y: test_y}
                test_acc_temp, test_loss_temp = sess.run([accuracy_mlo, cost_mlo], feed_dict=feed_dict_model)
                test_loss.append(test_loss_temp)
                test_accuracy.append(test_acc_temp)
        overall_test_accuracy = np.mean(test_accuracy)
        overall_test_loss = np.mean(test_loss)
        print(" Test Accuracy: ", overall_test_accuracy, " Test Loss: ", overall_test_loss)

    summary_writer.close()

    savePath = saver.save(sess, model_save)