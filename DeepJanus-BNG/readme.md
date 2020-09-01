# Test Input Generator for BeamNG #

## General Information ##
This folder contains the application of the DeepJanus approach to the steering angle prediction problem.
This tool is developed in Python on top of the DEAP evolutionary computation framework. It has been tested on a Windows machine equipped with a i9 processor, 32 GB of memory, and an Nvidia GPU GeForce RTX 2080 Ti with 11GB of dedicated memory.

## Dependencies ##

This tool needs the BeamNG simulator to be installed on the machine where it is running. 
A free version of the BeamNG simulator for research purposes can be found at https://beamng.gmbh/research/. 
We tested our tool only with the unlimited version of the simulator that has been provided to us directly by BeamNG GmbH.

To easily install the dependencies with pip, we suggest to create a dedicated virtual environment and run the command:

```pip install -r requirements.txt```

Otherwise, you can manually install each required library listed in the requirements.txt file using pip.

_Note:_ the version of Shapely should match your system.

## BeamNG Simulator Installation ##

0. Install a Subversion (SVN) client, e.g. TortoiseSVN.
1. Go to the directory where you want to checkout the BeamNG.research repository.
2. SVN Checkout the version 1.6.1.0 of the simulator (the one we used for the experiments);
3. Provide the URL of the repository that is either:
3.1 The URL of the standard version is https://projects.beamng.com/svn/research. 
**Note:** we have tested our tool with the unlimited version instead. 
3.2 The URL of the unlimited version (that we used for our experiments) is https://projects.beamng.com/svn/beamng-research_unlimited/ 
**Note:** you will be prompted to add your username and password to checkout the unlimited version. These may be obtained by sending a mail to research@beamng.gmbh and providing your motivation.
4. Add an environment variable named BNG_HOME pointing to the folder **trunk** (the full path), which is inside the repository you checked out. 
5. Edit the Path environment variable by adding %BNG_HOME%.

The less powerful graphics card we have successfully tested our tool with is an NVIDIA GeForce 940MX.

## Usage ##

### Input ###

* A trained model in h5 format. The default one is in the folder _data/trained_models_colab_;
* The seeds used for the input generation. The default ones are in the folder _data/member_seeds_;
* _core/config.py_ and _self_driving/beamng_config.py_ containing the configuration of the tool selected by the user.

### Output ###
When the run is finished, the tool produces the following outputs in the folder specified by the user:
* _config.json_ reporting the configuration of the tool;
* _report.json_ containing the final report of the run;
* the folder _archive_ containing the generated inputs (both the data structure in json and the image representation in svg format).

### Run the Tool ###
Run _self_driving/main_beamng.py_

## More Usages ##

### Train a New Predictor ###

* Run _udacity_integration/train-dataset-recorder-brewer.py_  to generate a new training set;
* Run _udacity_integration/train-from-recordings.py_  to train the ML model.

### Generate New Seeds ###

Run _self_driving/main_beamng_generate_seeds.py_

## License ##
The folder _beamngpy_ contains a version we modified of the code from https://github.com/BeamNG/BeamNGpy that is also shared under the MIT license by its authors.
