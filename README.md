# Food Identification Models

With this project I aim to build a model that can be deployed and used to identify food. This is a common machine learning problem and I hope to understand the issue more closely, and being to build an intuition with Convolutional Neural Networks (CNN).

## Getting Started

Open Jupyter Notebook in this folder and open up the setup.ipynb. The notebook will go through steps for setup and installation of the proect.

### Prerequisites

* Anaconda Environment
* jupyter notebook
* scikit-learn
* numpy
* Pandas
* matplotlib
* imutils
* cv2 (to be removed for a lighter package)
* tensorflow 2.0
* pytorch


### Steps to getting setup
1. Download the data.
2. Extract Contents from '.tar' file into root of project folder.
3. Rename the folder "food-101" to "data".
3. Open up the [Exploratory Data Analysis Notebook](exploratory_data_analysis.ipynb) and let all of the cells run. This will arrange the data and do some image pre-processing.
4. Open up any of the following notebooks and check out how the model or algorithm did:  
    a. [Logistic Regression](logistic_regression.ipynb)  
    b. [KNN Classifier](KNN_classifier.ipynb)  
    c. [CNN](CNN.ipynb)  
    d. [Pre-trained VGG16 with Custom End Layers](CNN_VGG16.ipynb)  



docker build --tag food-class $(pwd)/app
docker run -p 5000:5000 -v /$(pwd)/app:/app food-class