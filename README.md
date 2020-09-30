<h1 align="center">
  TASK 1
</h1>

## Setup
The file `microservices/docker_vm_ip.json` contains the docker VM local IP Address. If the docker VM is set to bridged local network or host only it will have its own local IP. 
Default Docker VM is localhost. My Docker VM IP is 192.168.99.100. Please change according to your requirement.

Execute the setup.sh file 

## Overview
Upon completition of setup three docker container are spun up with images docker_eder_postgress_image_landing, 
docker_eder_postgress_image_target, docker_eder_miroservices_image.

### docker_eder_postgress_image_landing
This conatiner runs ubuntu installed with postgress the setup.sh will start the container and bind the ports 5432 of host to 5432 of the container.
This container will hold the datasets which will be fed to the models in the microservice.

### docker_eder_postgress_image_target
This conatiner runs ubuntu installed with postgress the setup.sh will start the container and bind the ports 5434 of host to 5432 of the container.
This container will hold the results which are written by the microservice.

### docker_eder_miroservices_image
This container runs python3 installed with flask, scikit-learn, numpy, pandas and psycopg2. The host post 5000 is binded to the container port 5000.

### microservice apis

Run the /load_dataset_to_db first to load the data into landing db.

The microservice has three apis:

    [1]/load_dataset_to_db
       [GET]
       response:
          done
    
    [2] /predict_single
     request sample :
     [GET]
         Headers:
         Content-Type: application/json
         Body:
          {
          "Pclass" : 3, 
          "Sex" : 0, 
          "Age" : 2, 
          "Fare" : 0, 
          "Cabin" : 2, 
          "Embarked" : 2, 
          "Title" : 0, 
          "FamilySize" : 0,
          "Model" : "SVC"
          }
      Respose :
          {
          "msg": "prediction sucessful",
          "attributes": {
              "Pclass": 3,
              "Sex": 0,
              "Age": 2,
              "Fare": 0,
              "Cabin": 2,
              "Embarked": 2,
              "Title": 0,
              "FamilySize": 0
          },
          "model": "SVC",
          "prediction": 0
        }
        
        
    [3] /predict_batch
      request sample :
      [GET]
         Headers:
            Content-Type: application/json
         Body:
            {
              "psg_start" : 1000,
              "psg_end" : 1010,
              "model" : "SVC"
            }
       response :
         done
       comments:
         The predictions is loaded into the target table along with timestamp of when it was predicted.
    
