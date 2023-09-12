# Building docker environment
You are need to setup nvidia docker (version 2.0) for faster and easier executing of code:
[Installation guide](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)/)

After nvidia docker was correctly installed build docker image from megogo_final directory
```
docker build -t megogo2019:gpu ./docker
```
Make volume for using inside docker container, make sure, that you're redefine device parameter to proper path
```
docker volume create --name MEGOGO2019 --opt type=none --opt device=path/to/megogo_final/ --opt o=bind
```
# Evaluate on pretrained models
Making submission based on previously pretrained models was tested on such computer spec:
  -cpu: i7-7800X
  -gpu: GTX 1080 Ti
  -ram: 64Gb + 128Gb swap
Approximate time for executing is 5 minutes
```
docker run -v MEGOGO2019:/MEGOGO2019 --runtime=nvidia --name MEGOGO2019 -t megogo2019:gpu bash -c 'cd /MEGOGO2019/scripts/; bash ./run_evaluate.sh'
```
Result archive was placed under megogo_final/submissions folder
# Traininig and evaluation
Computer spec same as at evaluation section
Approximate time for full execution of training and evaluation process is 3-4 hours
```
docker run -v MEGOGO2019:/MEGOGO2019 --runtime=nvidia --name MEGOGO2019 --shm-size=64g -t megogo2019:gpu bash -c 'cd /MEGOGO2019/scripts/; bash ./run_training_evaluate.sh'
```
