# IAssistant
An assistant for the doppelmarsh DVIC Project

There are 3 parts to this repo. Each are adapted from two models
the BiLSTM from and mpnet based on https://huggingface.co/microsoft/mpnet-base.

## 1 - The training

### - BiLSTM
### - trainingagainmicrosoft


## 2- The pretraining
###   - pretrainedBiLSTM.py
###   - pretrainedmpnet.py
###   - pretrainedBiLSTM.ipynb

## 3 The API
###   - build/api folder
###   - try it on dockerhub : 

## 1 - The training 
Here for each model we train and save the models to be used in the pretraining phases
## 2 - The pretraining
Here we need to load the models before being able to 
## 3 - The API
Here we use the models to make predictions 

There is a dockerfile to build the image from these predictions
To prepare the docker image, run the following command from the root of the repository:
    
    ```bash
    make build
    make run
    "insert your docker container name here"
    ```
