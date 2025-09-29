# IFCBClassify_DEMO

## SETUP
Handy guide to getting started with PyTorch:

https://pytorch.org/get-started/locally/

First make sure you have Python installed on your machine. Any version of Python compatible with PyTorch should work just fine (at time of writing this was 3.9 or later).


Secondly, if you have an Nvidia GPU enabled machine (strongly advised) then make sure you have the CUDA toolkit installed. This will mean PyTorch can use your GPUs to speed up training.

https://developer.nvidia.com/cuda-toolkit

(Recommended) Install an Integrated Development Environment to edit and run the code. I suggest Visual Studio Code but many others are available.

https://code.visualstudio.com/

Finally setup a Python virtual environment. Whilst not essential it is generally a good idea to setup a virtual environment when working in Python. This keeps you installed libraries in a self contained sandbox separate from any other Python programs you may be working on.

### setup a virtual environment

(NOTE: if using VS Code you can CTRL + SHIFT + P - choose Python Interpreter > Create Virtual Environment to get VSCode to do these next steps for you.)

In a command prompt navigate to the folder where you unpacked the project.

'cd /path/to/project/'

Setup a virtual environment using Python venv:

`python -m venv .venv`

This will install a virtual environment in a sub folder called .venv in this project (you could actually setup a virtual environment anywhere else on your machine but I prefer to keep my projects together so just use a subfolder called .venv). 

Activate the VENV

Windows - 

`.venv\Scripts\activate.bat`

Linux - 

'source .venv/bin/activate'

## INSTALL REQUIRED PACKAGES


Install all the required packages (except PyTorch):

`pip install -r requirements.txt`

Next use the Install PyTorch tool on pytorch.org to generate the command you need to install PyTorch. If you have CUDA installed make sure to select the version which supports CUDA.

## GET STARTED

You should now be able to run the code in IFCBClassify_TRAIN.ipynb and build your first classifier. Some sample training and testing data is included in the project to get you started but you should use your own as soon as you are ready.
