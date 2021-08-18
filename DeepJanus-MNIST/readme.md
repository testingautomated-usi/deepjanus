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
