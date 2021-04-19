# AlphaBuilding-MedOffice
This is the official repository of AlphaBuilding MedOffice: a realistic OpenAI Gym environment that can be used to train, test and benchmark controllers for medium size office (1AHU + 9VAV boxes)

## Structure

``analysis``: code to analyze the result

``docker``: scripts to develop the docker image

``lib``: DRL models and utility functions

``log``: log files for TensorBoard visualization

``RL``: scripts for DRL

``virtualEnv``: scripts to develop the virtual simulation environment

## Test

1. Pull the docker image: ``$ docker pull walterzwang/drl_eplus:9.2``
2. ``$ cd AlphaBuilding/docker/drl``, revise parameter VOLUMN_PATH in the makefile, then ``$ make run``
3. Run scripts:
    * test fmu: ``$ cd virtualEnv``, run``python fmuModel/test_fmu.py``
    * test gym: ``$ cd virtualEnv``, run``python test_gym.py``
4. Run DRL algorithms:
    * All scripts must be run from the root repository
