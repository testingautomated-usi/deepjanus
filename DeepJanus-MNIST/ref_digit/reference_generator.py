import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


def input_reshape(x):
    # shape numpy vectors
    x_reshape = x.reshape(x.shape[0], 28, 28, 1)
    x_reshape = x_reshape.astype('float32')
    x_reshape /= 255.0
    return x_reshape


gray = cv2.imread("cinque.png", cv2.IMREAD_GRAYSCALE)

# resize the images and invert it (black background)
gray = cv2.resize(255-gray, (28, 28))

(thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

while np.sum(gray[0]) == 0:
    gray = gray[1:]

while np.sum(gray[:,0]) == 0:
    gray = np.delete(gray,0,1)

while np.sum(gray[-1]) == 0:
    gray = gray[:-1]

while np.sum(gray[:,-1]) == 0:
    gray = np.delete(gray,-1,1)

rows,cols = gray.shape

if rows > cols:
    factor = 20.0/rows
    rows = 20
    cols = int(round(cols*factor))
    gray = cv2.resize(gray, (cols,rows))
else:
    factor = 20.0/cols
    cols = 20
    rows = int(round(rows*factor))
    gray = cv2.resize(gray, (cols, rows))


colsPadding = (int(np.math.ceil((28 - cols) / 2.0)), int(np.math.floor((28 - cols) / 2.0)))
rowsPadding = (int(np.math.ceil((28-rows)/2.0)),int(np.math.floor((28-rows)/2.0)))
gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')
gray = (np.expand_dims(gray, 0))

pixels = gray.reshape((28, 28))
plt.imshow(pixels, cmap='gray')
plt.show()

plt.imsave("cinque_rp.png", gray.reshape(28, 28), cmap='gray')

np.save("cinque_rp", input_reshape(gray))
