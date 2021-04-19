# README

This repo contains scripts to develop the virtual simulation environment

## structure

``eplus`` contains building models in the form of EnergyPlus idf files

``fmuModel`` contains the validated fmu models and test files


## workflow

All the local command is recommended to be run from `~/git/AlphaBuilding/virtualEnv`


### 1. Test the E+ model with E+

_Could only be done locally, but not from the docker container_

run in the terminal:
```
energyplus -w eplus/weather/SF_TMY3.epw 
           -p eplus/EPlusTestRunRes/ 
              eplus/eplus_fmuInput/v1_csv.idf
```
  - EPlus has been added to the ``$PATH variable`` (by editing ``bashrc``), and therefore 
    could be called from terminal in any directory
  - result will be saved in the folder E+TestRunRes; steps to read the result:
    1. save out.eso file to ~/EPlus/EnergyPlus-8-5-0/PostProcess

    2. make sure the file name is eplusout.eso (might need to add eplus at the begining)

    3. cd to ``~/EPlus/EnergyPlus-8-5-0/PostProcess``, run in the terminal:
       ``./ReadVarsESO``

    4. read the generated csv file

### 2. Convert the E+ model to fmu

_Could only be done locally, but not from the docker container_

run in the terminal:
```
python ../../../EPlus/EPlusToFMU/Scripts/EnergyPlusToFMU.py 
       -i ../../../EPlus/EnergyPlus-9-2-0/Energy+.idd 
       -w eplus/weather/SF_TMY3.epw
       eplus/eplus_fmuInput/v1_fmu.idf 
```
Save the generated fmu in the fmuModel folders 


### 3. Use pyfmi to call the E+ FMU, E+ FMU is the slave

_**Could be done both locally and from docker container**_

run in the terminal:
```
python fmuModel/test_fmu.py
