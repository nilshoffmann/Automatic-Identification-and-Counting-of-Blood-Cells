#!/bin/bash
# Create a new model in the BioModels ML database

# Create a folder containing data, code and Dockerfile to train the model
mkdir -p docker_train
cp Dockerfile.train docker_train/Dockerfile
cp environment.yml docker_train/environment.yml
cp -r cfg docker_train/
cp -r darkflow docker_train/
cp -r data docker_train/
cp -r preprocess docker_train/
cp *.{py,R,md,txt} docker_train/
cp flow docker_train/
cd docker_train
docker build . -t rbc_yolov3_model



# Create a folder containing the pre-trained model and the Dockerfile to run it