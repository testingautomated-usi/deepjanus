import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
import argparse
import os

from udacity_integration.batch_generator import Generator
from udacity_integration.udacity_utils import INPUT_SHAPE, batch_generator, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS

np.random.seed(0)


def load_data(args):
    """
    Load training data and split it into training and validation set
    """
    tracks = ['training_recordings']

    x = np.empty([0, 3])
    y = np.array([])
    for track in tracks:
        drive = os.listdir(track)
        for drive_style in drive:
            try:
                csv_name = 'driving_log.csv'
                csv_folder = os.path.join(track, drive_style)
                csv_path = os.path.join(csv_folder, csv_name)

                def fix_path(serie):
                    return serie.apply(lambda d: os.path.join(csv_folder, d))

                data_df = pd.read_csv(csv_path)
                pictures = data_df[['center', 'left', 'right']]
                pictures_fixpath = pictures.apply(fix_path)
                csv_x = pictures_fixpath.values

                csv_y = data_df['steering'].values
                x = np.concatenate((x, csv_x), axis=0)
                y = np.concatenate((y, csv_y), axis=0)
            except FileNotFoundError:
                print("Unable to read file %s" % csv_path)
                exit()

    try:
        X_train, X_valid, y_train, y_valid = train_test_split(x, y, test_size=args.test_size, random_state=0)
    except TypeError:
        print("Missing header to csv files")
        exit()

    print("Train dataset: " + str(len(X_train)) + " elements")
    print("Test dataset: " + str(len(X_valid)) + " elements")
    return X_train, X_valid, y_train, y_valid


def build_model(args):
    """
    Modified NVIDIA model
    """
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Dropout(args.keep_prob))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()

    return model


def train_model(model, args, X_train, X_valid, y_train, y_valid):
    """
    Train the model
    """
    os.makedirs('trained_models', exist_ok=True)
    checkpoint = ModelCheckpoint('trained_models/self-driving-car-{epoch:03d}-2020.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=args.save_best_only,
                                 mode='auto')

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate))

    train_generator = Generator(X_train, y_train, True, args)
    validation_generator = Generator(X_valid, y_valid, False, args)

    model.fit_generator(train_generator,
                        validation_data=validation_generator,
                        epochs=args.nb_epoch,
                        use_multiprocessing=False,
                        max_queue_size=10,
                        workers=4,
                        callbacks=[checkpoint],
                        verbose=1)
    # use_validation_multiprocessing


def s2b(s):
    """
    Converts a string to boolean value
    """
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'


def main():
    """
    Load train/validation data set and train the model
    """
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory', dest='data_dir', type=str, default='.')
    parser.add_argument('-t', help='test size fraction', dest='test_size', type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability', dest='keep_prob', type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs', dest='nb_epoch', type=int, default=200)
    # parser.add_argument('-s', help='samples per epoch', dest='samples_per_epoch', type=int, default=100)
    parser.add_argument('-b', help='batch size', dest='batch_size', type=int, default=128)
    parser.add_argument('-o', help='save best models only', dest='save_best_only', type=s2b, default='true')
    parser.add_argument('-l', help='learning rate', dest='learning_rate', type=float, default=1.0e-4)
    args = parser.parse_args()

    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    data = load_data(args)
    model = build_model(args)
    train_model(model, args, *data)


if __name__ == '__main__':
    main()
