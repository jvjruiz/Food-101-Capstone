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

### Steps to Running API
1. Build the container with the following command: `docker build --tag food-class $(PATH TO ROOT OF PROJECT)/app`
2. Run Container with the following command `docker run -p 5000:5000 -v /$(PATH TO ROOT OF PROJECT)/app:/app food-class`
3. Open up your browser and go to http://localhost:5000 to access the webapp



