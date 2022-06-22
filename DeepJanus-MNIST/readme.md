# Test Input Generator for MNIST - Getting Started #

## General Information ##
This folder contains the application of the DeepJanus approach to the handwritten digit classification problem.
The following instructions allow to rapidly run DeepJanus, without configuring your environment from scratch.

> NOTE: If you want to configure your machine to run DeepJanus-MNIST, please read our [__detailed installation guide__](FULL_INSTALL.md)

## Step 1: Configure the environment  ##

Pull our pre-configured Docker image for DeepJanus-MNIST:

``` 
docker pull p1ndsvin/ubuntu:dj
```

Run the image by typing in the terminal the following commands:

```
docker run -it --rm p1ndsvin/ubuntu:dj
cd venvs
. .djvenv/bin/activate
cd ..
```

## Step 2: Run DeepJanus ##
Use the following commands to start a rapid run of DeepJanus-MNIST:

```
cd deepjanus/DeepJanus-MNIST
python main.py
```

## Usage ##

### Input ###

* A trained model in h5 format. The default one is in the folder `models`;
* A list of seeds used for the input generation. The default list is in the folder `original_dataset`;
* `config.py` containing the configuration of the tool selected by the user.

### Output ###
When the run is finished, the tool produces a folder named `run_x` (where `x` is the timestamp of the run) located in the folder `runs`. The output folder contains
the following outputs:
* `config.json` reporting the configuration of the tool;
* `stats.csv` containing the final report of the run;
* the folder `archive` containing the generated inputs (both in npy array and image format).

## Troubleshooting ##

* If tensorflow cannot be installed successfully, try to upgrade the pip version. Tensorflow cannot be installed by old versions of pip. We recommend the pip version 20.1.1.
* If the import of cairo, potrace or other modules fails, check that the correct version is installed. The correct version is reported in the file requirements.txt. The version of a module can be checked with the following command:
```
$ pip show modulename | grep Version
```
To fix the problem and install a specific version, use the following command:
```
$ pip install 'modulename==moduleversion' --force-reinstall
```

# 2. Attetion Maps Feature - Getting Started #
All the changes were made on the docker image using the same components installed in the Venv. Run Step1 to configure the enviroment.
The only new component to be installed is the tf-keras-vis. That can be installed running the following command:
## Step 2.1: Install   ##

```
$ pip install tf-keras-vis
```

## Usage ##
A new python file `attention_maps.py` was added to the directory. Inside this file, the main important functions to be used in the DeepJanus project are:
* `AM_get_attetion_svg_points_images_mth1`: This function will return a list containing the SVG path points located inside the square patch with more attention (sum of the attetion pixels inside the square patch). First, the function go through all the image summing the pixels inside the square patches and looking for the highest sum value. When it finds that area with maximum value, it will get all the SVG path points inside that area and return them in a list. 
  * ### Inputs: ###
    * `images`: A numpy array of the image to be processed with dimensions (x, 28, 28) where x>=1;
    * `x_patch_size`: The X size of the patch area for the sum of the attention pixels.
    * `y_patch_size`: The Y size of the patch area for the sum of the attention pixels.
    * `svg_path`: A string with the digit's SVG path description. Ex: "M .... C .... Z".
  * ### Outputs: ###
    * `list_of_ControlPointsInsideRegion`: A list containing the positions (tuples) of the SVG path points inside the max attention square patch.
    * `Elapsed time`: Elapsed time to run the function.
  * ### Illustration: ###
<img src="imgs/mth1.png" width="400"></img>

* `AM_get_attetion_svg_points_images_mth2`: This function will return a list containing the SVG path points and the respective weights for the random choice of a number in a non-unifiform distribution. Differently from the previous function, the sum of the attention pixels is performed only around the positions of the SVG path points (square patch). After saving the value of the attention maps for each SVG path point, the script will associate a weight for each point proportional to the value of the attention sum around their respective positions.
  * ### Inputs: ###
    * `images`: A numpy array of the image to be processed with dimensions (x, 28, 28) where x>=1
    * `sqr_size`: The size of the square patch which the sums of the attention pixels will be performed. Must be 3 or 5.
    * `model`: The model object to be used to predict to predict the digit's value.
  * ### Outputs: ###
    * `list_of_points_and_weights`: A list containing the positions (tuples) of the SVG path points and the respective non-uniform distribution weights.
    * `Elapsed time`: Elapsed time to run the function.
  * ### Illustration: ###
<img src="imgs/mth2.png" width="400"></img>

* ### Examples: ###
```python
#------------Example how to use------------#

from tensorflow import keras
from config import MODEL
from attention_maps import AM_get_attetion_svg_points_images_mth1, \
    AM_get_attetion_svg_points_images_mth2, \
    get_svg_path

# load the MNIST dataset
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

model = keras.models.load_model(MODEL)

images = x_test[:2]
svg_path = get_svg_path(images[0]) #get_svg_path input should be an image (28,28)

print("Method1:\n")
list_of_points_inside_square_attention_patch, elapsed_time = AM_get_attetion_svg_points_images_mth1(images, 3, 3, svg_path)
print(list_of_points_inside_square_attention_patch,"\n", elapsed_time,"\n")

print("Method2:\n")
list_of_points_and_probalities, elapsed_time = AM_get_attetion_svg_points_images_mth2(images, 3, model)
print(list_of_points_and_probalities,"\n", elapsed_time,"\n")

````

