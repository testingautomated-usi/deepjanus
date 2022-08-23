import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

csv_file = "./mutants/stats_3.csv"
df_number_of_missClassifications = pd.read_csv(csv_file, usecols=["#MissClass_found_att", "#MissClass_found_Normal"])

number_of_MC_att = df_number_of_missClassifications["#MissClass_found_att"]
print(number_of_MC_att.values)
print(number_of_MC_att.values.shape[0])

number_of_MC_normal = df_number_of_missClassifications["#MissClass_found_Normal"]
print(number_of_MC_normal.values)

x_data = np.vstack((number_of_MC_att.values, number_of_MC_normal.values))
print("x_data", x_data)
print(x_data.shape)
print(x_data.reshape(50,2))

print(df_number_of_missClassifications.values)

colors = ['blue', 'red']
labels = ["Attention Method", "Normal Method"]
num_bins = 12
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
plt.xticks(range(6))
plt.yticks(range(51))
plt.grid(True)
title = "Histogram for N= " + str(df_number_of_missClassifications.index.shape[0]) + "\n"
plt.title(title,
          fontweight ="bold")




plt.savefig("histogram.png")
plt.cla()
plt.clf()
