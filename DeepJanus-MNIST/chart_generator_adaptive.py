import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

csv_file = "./stats_3.csv"
df_number_of_missClassifications = pd.read_csv(csv_file, usecols=["#MissClass_found_att", "#MissClass_found_att_adaptive", "#MissClass_found_Normal"])
N = df_number_of_missClassifications.index.shape[0]
number_of_repetitions = 5
number_of_methods = 3

#ATT PART
number_of_MC_att = df_number_of_missClassifications["#MissClass_found_att"]
print(number_of_MC_att.values)
print(number_of_MC_att.values.shape[0])

#ADAPTIVE PART
number_of_MC_att_adaptive = df_number_of_missClassifications["#MissClass_found_att_adaptive"]
print(number_of_MC_att_adaptive.values)
print(number_of_MC_att_adaptive.values.shape[0])

#NORMAL PART
number_of_MC_normal = df_number_of_missClassifications["#MissClass_found_Normal"]
print(number_of_MC_normal.values)

x_data = np.vstack((number_of_MC_att.values, number_of_MC_att_adaptive.values, number_of_MC_normal.values))
print("x_data", x_data)
print(x_data.shape)
print(x_data.reshape(N,number_of_methods))

print(df_number_of_missClassifications.values)

colors = ['blue', 'green', 'red']
labels = ["Attention Method", "Att Adaptive Method", "Normal Method"]
num_bins = (number_of_repetitions + 1) * number_of_methods
n, bins, patches = plt.hist(df_number_of_missClassifications.values, num_bins, 
         histtype ='bar',
         color = colors,
         label = labels )
# plt.plot(bins, [0,1,2,3,4,5])

plt.tick_params(axis='y', which='major', labelsize=5)
plt.tick_params(axis='y', which='minor', labelsize=2)
plt.legend(prop ={'size': 10})

plt.xlabel("Number of Miss Classifications Found")
plt.ylabel("Number of Initial Digits")
plt.xticks(range(number_of_repetitions + 1))
plt.yticks(range(N + 1))
plt.grid(True)
title = "Histogram for N= " + str(N) + "\n"
plt.title(title,
          fontweight ="bold")




plt.savefig("histogram.png")
plt.cla()
plt.clf()
