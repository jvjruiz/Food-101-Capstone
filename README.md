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
3. Open up the [Exploratory Data Analysis Notebook](../notebooks/exploratory_data_analysis.ipynb) and let all of the cells run. This will arrange the data and do some image pre-processing.
4. Open up any of the following notebooks and check out how the model or algorithm did:  
    a. [Logistic Regression](logistic_regression.ipynb)  
    b. [KNN Classifier](KNN_classifier.ipynb)  
    c. [CNN](CNN.ipynb)  
    d. [Pre-trained VGG16 with Custom End Layers](CNN_VGG16.ipynb)  

### Steps to Running API
1. Build the container with the following command:
2. Run Container with the follow command
`docker build --tag food-class $(PATH TO ROOT OF PROJECT)/app`
3. Open up your browser and go to http://localhost:5000 to access the webapp
`docker run -p 5000:5000 -v /$(PATH TO ROOT OF PROJECT)/app:/app food-class`



