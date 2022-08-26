# RL_SelfDriving_Cars_Bap_Thesi
You only need 2 python files to run the program main.py and rl.py

The only FMU enviorments you should use is the ACCTestBenchExampleNoiNoise and ACCTestBenchExampleWithNoise. 

These contain the final FMU enviorments, one with white noise addition into to output signals (observation signals) and one without any noise addition

The .csv files are excel files that contain the parameters used within the hyperparameter tuning trials

FinalTests, Logs and test folder contains the events that are used in tensorflow.



---------------------------------------------------------------IMPORTANT----------------------------------------------------------------------------------

MAKE SURE TO CHANGE THE FMU PATH TO THE RIGHT FMU PATH AT LINES 39 AND 40 IN RL.PY
THIS MEAN USE accFMUPath = ".\ACCTestBenchExampleNoNoise.fmu" IF YOU TO USE THE NOISELESS ENVIRONMENT
AND USE #accFMUPath = ".\ACCTestBenchExampleWithNoise.fmu" IF YOU WANT TO USE AN ENVIORNMENT WITH NOISE ADDITION
------------------------------------------------------------------------------------------------------------------------------------------------------------





---------------------------------------------------------------IMPORTANT----------------------------------------------------------------------------------
MAKE SURE TO CHANGE LINES 310 AND 311 IN MAIN.PY TO YOUR DESIRES
IF YOU WANT TO USE HYPERPARAMETERS SET useHyperParameters = 1 IF NOT SET IT TO 0
IF YOU WANT TO GENERATE A NEW TRAINED AGENT SET makeNewModel = 1 IF NOT SET IT TO 0. SETTING THIS TO 0 MEANS THE PROGRAM WILL DIRECTLY USE A TRAINED AGENT ZIP TO RUN THE TEST INSTEAD OF TRAINING A NEW ONE FIRST AND THEN RUNNING THE TEST
------------------------------------------------------------------------------------------------------------------------------------------------------------
