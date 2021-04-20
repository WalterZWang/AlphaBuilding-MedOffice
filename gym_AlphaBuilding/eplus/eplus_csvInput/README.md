# README

This repo contains the validated EPlus models with csv as inputs

## structure

``in.idf`` original model with ASHRAE Guideline controls

``in_AHU.idf`` the AHU outlet temp is controlled by csv inputs

``in_uncontrolled.idf`` the AHU outlet temp; supply air flow rate of one terminal is controlled by csv inputs

``in_uncontrolled_reheat.idf`` the AHU outlet temp; supply air flow rate and reheat of one terminal is controlled by csv inputs

``in_uncontrolled_reheat_all.idf`` the AHU outlet temp; supply air flow rate and reheat of all terminals in Mid level is controlled by csv inputs

``in_uncontrolled_reheat_all_noOB.idf`` the AHU outlet temp; supply air flow rate and reheat of all terminals in Mid level is controlled by csv inputs; occ, lig, and mels schedules replaced by Schedule:compact

``OccSimulator_out_IDF.csv``, ``sch_light.csv``, ``sch_MELs.csv`` random schedule of occ, lighting and mels generated from OS measure

``input_file.csv`` inputs of AHU outlet temp, terminal flow rate and reheat

## changes between versions
``in_uncontrolled_reheat_all_noOB.idf``: revised from ``in_uncontrolled_reheat_all.idf``
1. Delete all occ, lig and mels schedules from schedule:file objects
2. Add occ, lig and mels schedule:compact
  * we did not set different schedules for different zone spaces
3. replace the schedule in people, lights, electricequipment objects end with (X_sch)
  * only replace schedules for confRoom, opOff, and enOff
4. replace the schedule in EnergyManagementSystem:Sensor objects (only occ and equip sch)
