# Experimental Evaluation: Data and Scripts #

## General Information ##

This folder contains the data we obtained by conducting the experimental procedure described in the paper (Section 6).
It is structured as follows:
* [__evaluation_BNG__](/submissions/available/submission441/experiments/evaluation_BEAMNG) contains the data and the scripts of the experiment on BeamNG;
* [__evaluation_DLF__](/submissions/available/submission441/experiments/evaluation_DLF) contains the data and the scripts of the experiment on DLFuzz;
* [__evaluation_MNIST__](/submissions/available/submission441/experiments/evaluation_MNIST) contains the data and the scripts of the experiment on MNIST.
* [__results__](/submissions/available/submission441/experiments/results.xlsx) is a spreadsheet containing the results of the experiments, this is the data source for Tables 2, 3, 4 in the manuscript.

_Note:_ each sub-package contains further specific instructions.

## Dependencies ##

We ran the scripts in this folder with Python v3.
To easily install the dependencies with pip, we suggest to create a dedicated virtual environment and run the command:

`pip install -r requirements.txt`

Otherwise, you can manually install each required library listed in the requirements.txt file using pip.
