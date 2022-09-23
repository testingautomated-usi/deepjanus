import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from config import NUMBER_OF_MUTATIONS, NUMBER_OF_REPETITIONS, START_SEED, N,\
    HISTOGRAM1_PATH,\
    HISTOGRAM2_PATH,\
    HISTOGRAM3_PATH
import matplotlib.gridspec as gridspec

csv_file = "./mutants/stats_3.csv"
df_number_of_missClassifications = pd.read_csv(csv_file, usecols=["#MissClass_found_att", "#MissClass_found_att_adaptive", "#MissClass_found_Normal",  "#MissClass_found_Normal_adaptive"])

class histogram:
    def __init__(self,csv_path) -> None:
        self.csv_path = csv_path

    def get_columns(self, usecols):
        self.data_frame = pd.read_csv(self.csv_path, usecols=usecols)
        self.data_size = self.data_frame.index.shape[0]

    def set_colors(self, colors):
        self.colors = colors
    
    def set_labels(self, labels):
        self.labels = labels
    
    def set_title(self, title):
        plt.title(title,
                fontweight ="bold", fontsize=10)

    def generate_plot(self, subplot, bars_range=(0,5), histtype ='bar'):
        n, bins, patches = subplot.hist(self.data_frame.values,
            range = bars_range,
            histtype = histtype,
            color = self.colors,
            label = self.labels )

        plt.tick_params(axis='y', which='major', labelsize=5)
        plt.tick_params(axis='y', which='minor', labelsize=2)
        plt.legend(prop ={'size': 10})

        plt.xlabel("Number of Miss Classifications Found")
        plt.ylabel("Number of Initial Digits")
        plt.xticks(range(1, NUMBER_OF_REPETITIONS + 1))
        plt.yticks(range(self.data_size + 1))
        plt.grid(True)
            
    def save_histogram(self, path):
        plt.savefig(path)
        plt.cla()
        plt.clf()

def col_total_sum(DataFrame, col_ID):
    values_list  = DataFrame[col_ID]
    return np.sum(values_list.values)   

def plot_histograms_top_and_bottom(usecols, colors, labels, out_path, top_indices, bottom_indices):

    fig = plt.figure(figsize=(9,10))
    gs = gridspec.GridSpec(nrows=2,ncols=1, width_ratios=[1], height_ratios=[1,1])

    #Plotting comparison: Att vs Att Adaptive (Top and Bottom)
    
    top = fig.add_subplot(gs[0,0])
    title = "Histogram for N= " + str(N) + "\n#Mut= " + str(NUMBER_OF_MUTATIONS) + " / #Repetitions=" + str(NUMBER_OF_REPETITIONS) + "\nStart Seed=" + str(START_SEED)

    labels_top = [labels[top_indices[0]] + str(col_total_sum(df_number_of_missClassifications, usecols[top_indices[0]])),
                  labels[top_indices[1]] + str(col_total_sum(df_number_of_missClassifications, usecols[top_indices[1]]))]

    labels_bottom = [labels[bottom_indices[0]] + str(col_total_sum(df_number_of_missClassifications, usecols[bottom_indices[0]])),
                     labels[bottom_indices[1]] + str(col_total_sum(df_number_of_missClassifications, usecols[bottom_indices[1]]))]

    hist_top = histogram(csv_file)    
    hist_top.get_columns([usecols[top_indices[0]], usecols[top_indices[1]]])
    hist_top.set_colors([colors[top_indices[0]], colors[top_indices[1]]])
    hist_top.set_labels(labels_top)
    hist_top.set_title(title)
    hist_top.generate_plot(top, bars_range=(1,5))
    
    bottom = fig.add_subplot(gs[1,0])

    hist_bottom = histogram(csv_file)    
    hist_bottom.get_columns([usecols[bottom_indices[0]], usecols[bottom_indices[1]]])
    hist_bottom.set_colors([colors[bottom_indices[0]], colors[bottom_indices[1]]])
    hist_bottom.set_labels(labels_bottom)
    hist_bottom.generate_plot(bottom, bars_range=(1,5))

    hist_bottom.save_histogram(out_path)

if __name__ == "__main__":    

    #Plotting Plotting histogram comparison: four methods same image

    print("Plotting histogram comparison: four methods same image.")
    hist_1 = histogram(csv_file)
    usecols=["#MissClass_found_att", "#MissClass_found_att_adaptive", "#MissClass_found_Normal",  "#MissClass_found_Normal_adaptive"]
    hist_1.get_columns(usecols)
    colors = ['blue', 'green', 'red', '#CCCC00']
    labels = ["Attention Method= ", "Att Adaptive Method= ", "Normal Method= " , "Normal Adpt Method= "]

    title = "Histogram for N= " + str(N) + "\n#Mut= " + str(NUMBER_OF_MUTATIONS) + " / #Repetitions=" + str(NUMBER_OF_REPETITIONS) + "\nStart Seed=" + str(START_SEED)

    hist_1.set_colors(colors)
    hist_1.set_labels(labels)
    hist_1.set_title(title)
    hist_1.generate_plot(plt, bars_range=(1,5))
    hist_1.save_histogram(HISTOGRAM1_PATH)

    fig = plt.figure(figsize=(9,10))
    gs = gridspec.GridSpec(nrows=2,ncols=1, width_ratios=[1], height_ratios=[1,1])
    print(HISTOGRAM1_PATH)

    #Plotting histogram comparison: Adaptive vs Without Adaptive (Top and Bottom)

    print("Plotting histogram comparison: Adaptive vs Without Adaptive (Top and Bottom).")
    plot_histograms_top_and_bottom(usecols, colors, labels, HISTOGRAM2_PATH, top_indices=[0,1], bottom_indices=[2,3])
    print(HISTOGRAM2_PATH)
    #Plotting histogram comparison: Att vs Normal (Top and Bottom)

    print("Plotting histogram comparison: Att vs Normal (Top and Bottom).")
    plot_histograms_top_and_bottom(usecols, colors, labels, HISTOGRAM3_PATH, top_indices=[0,2], bottom_indices=[1,3])
    print(HISTOGRAM3_PATH)

    print("Histograms generated successfully!")
