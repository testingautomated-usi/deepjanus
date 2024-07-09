import vectorization_tools, rasterization_tools
from tensorflow import keras as K
from keras.layers import Input
import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt
import ModelA

img_rows, img_cols = 32, 32
img_chn = 3

datasetLoc = '/home/vin/PycharmProjects/dola/DistributionAwareDNNTesting/SVHN_dx/dataset/test_32x32.mat'
test_data = loadmat(datasetLoc)
x_test = np.array(test_data['X'])
y_test = test_data['y']

# Normalize data.
x_test = np.moveaxis(x_test, -1, 0)
#x_test = x_test.astype('float32') / 255

#y_test[y_test == 10] = 0
#y_test = np.array(y_test)

# 22 is critical
index = 22
seed_image = x_test[index]
explabel = y_test[index]
#seed_image = x_test[5]

for channel in range(seed_image.shape[2]):
    channel_img = seed_image[...,channel]
    plt.imsave("channel"+str(channel)+".png", channel_img, cmap='gray')

grey = np.mean(seed_image, axis=2)
plt.imsave("channel_mean.png", grey, cmap='gray')

xml_desc = vectorization_tools.vectorize(grey)

rasterized = rasterization_tools.rasterize_in_memory(xml_desc)

#if np.any(rasterized):
#    xml_desc = vectorization_tools.vectorize(grey)
#    rasterized = rasterization_tools.rasterize_in_memory(xml_desc)

color_rasterized = np.repeat(rasterized, 3, axis=3)
print(color_rasterized.shape)

input_shape = (img_rows, img_cols, img_chn)

# define input tensor as a placeholder
input_tensor = Input(shape=input_shape)

model1 = ModelA.ModelA(input_tensor)
print(model1.name)

pred1 = model1.predict(color_rasterized)
label1 = np.argmax(pred1[0])
print("expected label: " + str(explabel))
print("predicted label: "+str(label1))


v = color_rasterized * 255.0
v = v.astype('uint8')
v = v.reshape(32, 32, 3)
plt.imsave("channel_mean_vectorized.png", v, cmap='gray')

print(v.shape)
exit()





