import tensorflow as tf
import config as config
import numpy as np
from numpy import matrix
import re
import glob
import pandas as pd
import sys
import cv2


def load_dataset(dataset_folder,  xlsx):
    """

    :param dataset_folder:
    :param xlsx:
    :return:
    """
    train_dataset_folder = dataset_folder
    file_name = xlsx
    list_of_patient = []
    df = pd.read_excel(io=file_name)
    patient_id = df['patient_id'].tolist()
    patient_side = df['side'].tolist()
    patient_view = df['view'].tolist()
    target_value = df["assessment"].tolist()

    for i in range(len(patient_id)):
        list_of_patient.append([patient_id[i], patient_side[i], patient_view[i], target_value[i]])

    array_images =[]
    array_target = []
    count = 0

    for i in range(len(list_of_patient)):
        key_word = str(str(list_of_patient[i][0]).split("_")[-1])
        target_element = str(list_of_patient[i][3])
        temp_path = train_dataset_folder + "_" + key_word + "_" + list_of_patient[i][1] + "_" + list_of_patient[i][
            2] + ".png"

        if str(list_of_patient[i][1]) == "RIGHT":
            try:
                temp_image = cv2.imread(temp_path, 0)
                if temp_image is None :
                    print("Image Not Available: ", temp_path)
                temp_image = cv2.resize(temp_image, (2000, 2600))
                temp_image = np.array(temp_image).reshape(1, 2000, 2600, 1)
                if temp_image is not None:
                    array_images.append([temp_image, list_of_patient[i][2]])
                    count += 1

            except Exception as error:
                print(error)

        if str(list_of_patient[i][1]) == "LEFT":
            try:
                temp_image = cv2.imread(temp_path, 0)
                temp_image = cv2.flip(temp_image, 1)
                temp_image = cv2.resize(temp_image, (2000, 2600))
                temp_image = np.array(temp_image).reshape(1, 2000, 2600, 1)
                if temp_image is not None:
                    array_images.append([temp_image, list_of_patient[i][2]])
                    count += 1
            except Exception as error:
                print(error)

        if temp_image is not None:
            target_element = int(target_element)
            if target_element == 0:
                train_y = matrix([[1, 0, 0]])
            if target_element == 1:
                train_y = matrix([[0, 1, 0]])
            if target_element == 2:
                train_y = matrix([[0, 0, 1]])
            array_target.append(train_y)

    return array_images, array_target

