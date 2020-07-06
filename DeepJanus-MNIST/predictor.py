from tensorflow import keras
import numpy as np

from properties import MODEL, EXPECTED_LABEL, num_classes


class Predictor:

    # Load the pre-trained model.
    model = keras.models.load_model(MODEL)
    print("Loaded model from disk")

    @staticmethod
    def predict(img):
        explabel = (np.expand_dims(EXPECTED_LABEL, 0))

        # Convert class vectors to binary class matrices
        explabel = keras.utils.to_categorical(explabel, num_classes)
        explabel = np.argmax(explabel.squeeze())

        # Predictions vector
        predictions = Predictor.model.predict(img)

        prediction1, prediction2 = np.argsort(-predictions[0])[:2]

        # Activation level corresponding to the expected class
        confidence_expclass = predictions[0][explabel]

        if prediction1 != EXPECTED_LABEL:
            confidence_notclass = predictions[0][prediction1]
        else:
            confidence_notclass = predictions[0][prediction2]

        return prediction1, confidence_expclass, confidence_notclass
