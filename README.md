# Food Identification Model

With this project I aim to build a model that can be deployed and used to identify food. This is a common machine learning problem and I hope to understand the issue more closely, and being to build an intuition with Convolutional Neural Networks (CNN). For this project I have chosen to build the model using PyTorch as my framework, and used transfer learning on a pre-trained ResNet50 model.

## Getting Started

### Prerequisites

* Anaconda Environment
* jupyter notebook
* scikit-learn
* numpy
* Pandas
* matplotlib
* pytorch
* pillow
* Flask
* requests
* torchvision

### Steps to getting setup
1. Download the data.
2. Extract Contents from '.tar' file into root of project folder.
3. Rename the folder "food-101" to "data".
3. Open up the [Exploratory Data Analysis Notebook](notebooks/pytorch_cnn_resnet50.ipynb) and let all of the cells run. 
    1. This will arrange the data and do some image pre-processing. 
    2. Start training Model
    3. Evaluate and visualize model

### Steps to Training Model
There are two ways to train the model with this repo.
1. Through the jupyter notebook
2. Directly though the python script

To run with the jupyter notebook follow these steps:
1. Run `jupyter notebook` in the command line
2. Navigate to the notebooks folder and open pytorch_cnn_resnet50.ipynb
3. Run all cells

To run directly with the scripts follow these steps:
1. `cd src/`
2. python train.py

### Steps to Running API\
1. Download pre-trained model [Here](https://drive.google.com/open?id=1pixiwUplEVh-ZjTtYERvec-ijsHIKby6)
2. Move downloaded model to the folder $(PROJECT ROOT)/app/models
2. Build the container with the following command: `docker build --tag food-class $(PROJECT ROOT)/app`
2. Run Container with the following command `docker run -p 5000:5000 -v /$(PROJECT ROOT)/app:/app food-class`
3. Open up your browser and go to http://localhost:5000 to access the webapp



