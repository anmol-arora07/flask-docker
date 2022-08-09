This repository uses docker to containerize a ML model such that it can be put into production
#####
####
####
Steps taking place in this repo-
1. train.py contains model training code (uses boston house data for a simple gradient boosting model)
2. Dockerfile builds the docker image and runs the train.py during image generation such that the docker image already contains serialized trained model for inference
3. Once the container is setup and running- we can execute api.py in the container to be able get real time prediction

Commands need to run through terminal-
1. To be build docker image which would train the model and save the serialized object in the image---     
 docker build -t docker-api -f Dockerfile .

2. To run the contain such that a REST endpoint is running----
 docker run -it -p 8000:8000 docker-api python3 api.py

3. To get prediction from the end point----
  curl -i -H "Content-Type: application/json" -X POST -d '{"CRIM": 15.02, "ZN": 0.0, "INDUS": 18.1, "CHAS": 0.0, "NOX": 0.614, "RM": 5.3, "AGE": 97.3, "DIS": 2.1, "RAD": 24.0, "TAX": 666.0,  "PTRATIO": 20.2, "B": 349.48, "LSTAT": 24.9}' 127.0.0.1:8000/predict


This can be deployed in production using kubernetes to scale the number of containers based on the load
// Will be adding it soon


