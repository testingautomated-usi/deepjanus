from tensorflow import keras

from distance_calculator import calc_angle_distance
from properties import MODEL, MISB_TSHD
import numpy as np


class Predictor:

    # Load the pre-trained model.
    model = keras.models.load_model(MODEL, compile=False)
    print("Loaded model from disk")

    @staticmethod
    def predict(img, head_pose, label):
        model = Predictor.model
        predictions = model.predict([img, head_pose])

        predictions1 = list()
        confidences = list()

        for i in range(len(predictions)):
            prediction1 = predictions[i]
            explabel = label[i]

            diff = calc_angle_distance(prediction1, explabel)
            diff = np.abs(np.degrees(diff))
            confidence = MISB_TSHD - diff

            predictions1.append(prediction1)
            confidences.append(confidence)

        return predictions1, confidences
