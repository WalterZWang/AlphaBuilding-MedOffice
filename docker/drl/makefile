IMG_NAME=walterzwang/drl_eplus:9.2
CONTAINER_NAME=drl_eplus92
VOLUMN_PATH=/home/user/Desktop/git/AlphaBuilding-MedOffice# to be revised: path to the repo

COMMAND_RUN=docker run \
           --name ${CONTAINER_NAME} \
		   --detach=false \
		   -e DISPLAY=${DISPLAY} \
		   -v /tmp/.X11-unix:/tmp/.X11-unix \
		   --rm \
		   -v ${VOLUMN_PATH}:/mnt/shared \
		   -i \
		   -t ${IMG_NAME} /bin/bash -c

build:
	docker build --no-cache --rm -t ${IMG_NAME} .

remove-image:
	docker rmi ${IMG_NAME}

run:
	$(COMMAND_RUN) "cd /mnt/shared && bash"
