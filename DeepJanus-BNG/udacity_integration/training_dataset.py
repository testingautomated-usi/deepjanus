import numpy as np
import os
import pandas as pd


class TrainingDataset:
    def __init__(self, root_folder='training_recordings'):
        self.root_folder = root_folder

    def load_all_datasets(self):
        tracks = [self.root_folder]

        x = np.empty([0, 3])
        y = np.array([])
        for track in tracks:
            drive = os.listdir(track)
            for drive_style in drive:
                try:
                    path = 'driving_log.csv'
                    csvpath = os.path.join(track, drive_style, path)

                    data_df = pd.read_csv(csvpath)
                    csv_x = data_df[['center', 'left', 'right']].values
                    csv_y = data_df['steering'].values
                    x = np.concatenate((x, csv_x), axis=0)
                    y = np.concatenate((y, csv_y), axis=0)
                except FileNotFoundError as ex:
                    print("Unable to read file %s" % path)
                    raise ex
        self.x = x
        self.y = y


if __name__ == '__main__':
    td = TrainingDataset()
    td.load_all_datasets()

    np.histogram(td.y)

    import numpy as np
    import matplotlib.mlab as mlab
    import matplotlib.pyplot as plt

    x = td.y

    n, bins, patches = plt.hist(x, 20, facecolor='blue', alpha=0.5)
    plt.show()
