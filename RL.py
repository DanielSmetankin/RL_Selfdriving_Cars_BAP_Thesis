import shutil
from typing import Any

import gym
import numpy as np
import optuna as optuna
import scipy
from fmpy import *
from fmpy import read_model_description, extract
from fmpy.fmi2 import FMU2Slave
from fmpy.util import plot_result, download_test_file
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


#import logging
#logger = logging.getLogger(__name__)
from scipy.constants import sigma
import optuna


def getVars(model_description):
    vrs = {}
    for variable in model_description.modelVariables:
        vrs[variable.name] = variable
    return vrs

def load(fmuPath, model_description, instanceName):
    unzipdir = extract(fmuPath)
    fmu = FMU2Slave(guid=model_description.guid,
                unzipDirectory=unzipdir,
                modelIdentifier=model_description.coSimulation.modelIdentifier,
                instanceName=instanceName)
    return fmu, unzipdir

#accFMUPath = ".\Acc01StepSizeB.fmu"
#accFMUPath = ".\ACCTestBenchExampleNoNoise.fmu"
accFMUPath = ".\ACCTestBenchExampleWithNoise.fmu"
episodeFactor = 100
START_TIME = 0.0
STOP_TIME = 34.0 # 80
STEP_SIZE = 0.1
TIMESTEPS = episodeFactor*int((STOP_TIME-START_TIME)/STEP_SIZE)

#Increase episodeFactor to bigger number for more episodes

# Note: safe_distance = distance_margine

model_description = read_model_description(accFMUPath)
dump(accFMUPath)

class FMIEnv(): # this is with normalization
    # https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['console']}  # human
    reward_range = (-float("inf"), float("inf"))
    spec = None

    # Set these in ALL subclasses
    action_space = None
    observation_space = None

    def __init__(self):
        super(FMIEnv, self).__init__()

      #  self.reward = reward
        # fault's value and time
        # to read: https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html ,, normilize (-1,1)
        # self.action_space = Box(low=np.array([0.0, 0.0]).astype(np.float32), high=np.array([5.0, STOP_TIME]).astype(np.float32))
        self.action_space = Box(low=np.array([-1.0, -1.0]).astype(np.float32),
                                     high=np.array([1.0, 1.0]).astype(np.float32))

        # acceleration, relative distance
        # self.observation_space = Box(low=np.array([-3.0, 0.0]).astype(np.float32), high=np.array([2.0, 200.0]).astype(np.float32))
        self.observation_space = Box(low=np.array([-1.0, -1.0, -1.0,-1.0]).astype(np.float32),
                                     high=np.array([1.0, 1.0, 1.0,1.0]).astype(np.float32))

        # Put variable refs in a dictionary
        self.VarsACC = getVars(model_description)

        # Load FMU
        self.accFMU, self.accFMUExtractDir = load(accFMUPath, model_description, "ACC")

        # initialize
        self.accFMU.instantiate(visible=False, callbacks=None)  # loggingOn=LOGGING
        self.accFMU.setupExperiment(startTime=START_TIME)
        self.accFMU.enterInitializationMode()
        self.accFMU.exitInitializationMode()

        self.accFMU.setReal([self.VarsACC['Acceleration'].valueReference], [-1.0])


        self.time = START_TIME
        self.rows = []  # list to record the results
        self.time_trajectory = []
        self.reward_trajectory = []
        self.acceleration_trajectory = []
        self.get_rel_dis_trajectory = []
        self.get_velocity_trajectory = []
        self.get_longi_velocity_trajectory = []
        self.safe_distance_trajectory = []
        self.jitterArray = []
        self.meanJitter = 0
        self.reset()
        self.teller = 0
        self.teller2 = 0




    def step(self, action):

        # Get the Observation signals (output signals)
        get_rel_dis = self.accFMU.getReal([self.VarsACC["relative_distance"].valueReference])[0]
        safe_distance = self.accFMU.getReal([self.VarsACC["safe_distance"].valueReference])[0]
        get_longi_velocity = self.accFMU.getReal([self.VarsACC["longitudinal_velocity"].valueReference])[0]

        if self.time <= 1.0:
            get_velocity = -2.2
        else:
            get_velocity = self.accFMU.getReal([self.VarsACC["relative_velocity"].valueReference])[0]


        # perform one step
        self.accFMU.doStep(currentCommunicationPoint=self.time, communicationStepSize=STEP_SIZE)


        # Apply action
        # Minimum Accelartion = -3
        # Maximum Accelartion = 2
        #  [x_min x_max] -> [a b] : x_normalized = a + ((x–x_min)*(b-a))/(x_max – x_min)
        set_acelaration = -3 + ((action[0] + 1) * 5) / 2  # [-1 1] -> [-3 2]   #faul_val should be Accelartion action signal this is where we scale it up
        self.accFMU.setReal([self.VarsACC['Acceleration'].valueReference], [set_acelaration])


        if self.time >= 0.2:
            jitter = (abs(self.acceleration_trajectory[self.teller-1] - self.acceleration_trajectory[self.teller - 2]))*STEP_SIZE
            if jitter == 0:
                jitter = 0.00001

        else:
            jitter = 0.0001
        self.jitterArray.append(jitter)

        """
        ---------------------------------------------------------------------
        """

        reward_selection = 2

        """
        ---------------------------------------------------------------------
        """

        if reward_selection == 1:

            #if len(self.acceleration_trajectory) > 1:
               # gaus_noise = np.random.normal(1, 0.1, set_acelaration.shape)
               # set_acelaration = set_acelaration + gaus_noise
            safte_coef = (get_rel_dis- safe_distance)*0.01


            if 100<=(get_rel_dis) and (get_rel_dis)  <=200:
                 sim_reward =  (set_acelaration)*0.013 + ((safte_coef)*0.0034)-jitter
            elif 70<=(get_rel_dis) and (get_rel_dis)  <100:
                sim_reward = (set_acelaration)*0.010 + ((safte_coef)*0.0044)-jitter
            elif 40 <= (get_rel_dis) and (get_rel_dis) < 70:
                sim_reward =  -(set_acelaration)*0.045 + ((safte_coef)*0.0054)-jitter
            elif 10 <= (get_rel_dis) and (get_rel_dis) < 40:
                sim_reward =  -(set_acelaration)*0.09 + ((safte_coef)*0.0064)-jitter
            elif 1 <= (get_rel_dis) and (get_rel_dis) < 10:
                sim_reward = -(set_acelaration) * 0.14 + ((safte_coef) * 0.54) - jitter
            elif 0 <= (get_rel_dis) and (get_rel_dis) < 1:
                sim_reward = -(set_acelaration) * 0.2 + ((safte_coef) * 0.64) - jitter
            elif (get_rel_dis - safe_distance) <= 0:
                sim_reward = -3
            else:
                sim_reward = 0



        elif reward_selection == 2:
            if(get_rel_dis<safe_distance):
                lead_car_velocity =get_longi_velocity- get_velocity
                vRef = min(lead_car_velocity, get_longi_velocity)
            else:
                vRef = get_longi_velocity

            velocity_error = vRef - get_longi_velocity

            if velocity_error**2 <= 5:
                Mt = 1
            else:
                Mt = 0

            if len(self.acceleration_trajectory)> 1:
                 sim_reward =-(0.1*velocity_error**2+self.acceleration_trajectory[self.teller-1]**2)+Mt  -3*jitter +(get_rel_dis- safe_distance)*0.1

            else:
                sim_reward = 0



        self.teller = self.teller + 1

        """
        if(get_rel_dis- safe_distance<0):
            self.teller2 = self.teller2 +1
            print("\n----------------------------------------------------------------------")
            print("relative distance ")
            print( get_rel_dis)
            print("\n safe distance ")
            print(safe_distance)
            print("\n teller ")
            print(self.teller2)
            print("\n time ")
            print(self.time)
            print("\n")
        """


        # Normalize The Observation Signals
        # [x_min x_max] -> [a b] : x_normalized = a + ((x–x_min)*(b-a))/(x_max–x_min)
       # acceleration_norm = -1 + ((acceleration + 3) * 2) / 5  # [-3 2] -> [-1 1]
        get_rel_dis_norm =  -1 +((get_rel_dis-0)*(2))/(200-0)        #  [0 200] -> [-1 1]
        get_rel_vel_norm =  -1 +((get_velocity+5)*(2))/(2.2+5)                      # [-5 2.2] --> [-1 1]
        get_safe_dist_norm = -1 + ((safe_distance -15) * (2)) / (50-15)           # [15 50] --> [-1 1]
        get_longi_velocity_norm  = -1 +((get_longi_velocity-0)*(2))/(22-0)    # [0 22] --> [-1 1]



        self.observation = np.array([ get_rel_dis_norm,get_rel_vel_norm,get_safe_dist_norm,get_longi_velocity_norm]).astype(np.float32)
        reward = sim_reward

        info = {'time':self.time, 'acceleFration': set_acelaration, 'Rel_dis': get_rel_dis}


        done = False
        if self.time >= STOP_TIME:
            self.meanJitter = np.mean(self.jitterArray)

            with open('meanJitter.txt', 'w') as f:
                f.write(str(self.meanJitter))
            self.accFMU.terminate()            # call the FMI API directly to avoid unloading the share library
            self.accFMU.freeInstance()
            # clean up
            shutil.rmtree(self.accFMUExtractDir, ignore_errors=True)

            done = True
            # info={'time': self.time_trajectory, 'reward': self.reward_trajectory,
            #         'Safe_dis': self.safe_distance_trajectory, 'Rel_dis': self.get_rel_dis_trajectory}
            print('Info:', info)
        else:
            done = False
            # Optionally we can pass additional info, we are not using that for now

        # Plot outputs
        self.time_trajectory.append(self.time)
        self.acceleration_trajectory.append(set_acelaration)
        self.get_longi_velocity_trajectory.append(get_longi_velocity)
        self.get_rel_dis_trajectory.append(get_rel_dis)
        self.get_velocity_trajectory.append(get_velocity)
        self.safe_distance_trajectory.append(safe_distance)
        self.reward_trajectory.append(sim_reward)


        # advance the time
        self.time = self.time + STEP_SIZE
        # sys.stdout.write("\rTime: %.3f" % (self.time))
        # sys.stdout.flush()

        # Return step information DON'T change the names below
        return self.observation ,reward, done, info  # return: (np.ndarray, float, bool, dict)






    def render(self, mode="console"):
        # Plot results
        # pass
     if(self.time>34.0):
        plt.subplot(4, 1, 1)
        plt.xlabel("time")
        plt.plot(self.time_trajectory, self.acceleration_trajectory, '-', label="Acceleration")
        plt.legend()

        plt.subplot(4, 1, 2)
        plt.xlabel("time")
        plt.plot(self.time_trajectory, self.get_rel_dis_trajectory, '-', label="relative distance",color='r')
        plt.legend()

        plt.subplot(4, 1, 2)
        plt.xlabel("time")
        plt.plot(self.time_trajectory, self.safe_distance_trajectory, '-', label="safe distance", color='g')
        plt.legend()

      #NP
        plt.subplot(4, 1, 3)
        plt.xlabel("time")
        plt.plot(self.time_trajectory, self.get_velocity_trajectory, '-', label="relative velocity" ,color='y')
        plt.legend()

        plt.subplot(4, 1, 3)
        plt.xlabel("time")
        plt.plot(self.time_trajectory, self.get_longi_velocity_trajectory, '-', label="actual velocity", color='b')
        plt.legend()

        plt.subplot(4, 1,4)
        plt.xlabel("time")
        plt.plot(self.time_trajectory, self.reward_trajectory, '-', label="reward")
        plt.legend()





        plt.show()

    def reset(self):
        # Put variable refs in a dictionary
        self.VarsACC = getVars(model_description)
        # Load FMU
        self.accFMU, self.accFMUExtractDir = load(accFMUPath, model_description, "ACC")

        # close model
        self.accFMU.terminate()
        # call the FMI API directly to avoid unloading the share library
        self.accFMU.freeInstance()
        # clean up
        shutil.rmtree(self.accFMUExtractDir, ignore_errors=True)

        # restart model
        self.VarsACC = getVars(model_description)
        # Load FMU
        self.accFMU, self.accFMUExtractDir = load(accFMUPath, model_description, "ACC")


        # initialize
        self.accFMU.instantiate(visible=False, callbacks=None)  # loggingOn=LOGGING
        self.accFMU.setupExperiment(startTime=START_TIME)
        self.accFMU.enterInitializationMode()
        self.accFMU.exitInitializationMode()


        self.accFMU.setReal([self.VarsACC['Acceleration'].valueReference], [-1.0])
       # self.accFMU.setReal([self.VarsACC['fault_time'].valueReference], [-1.0])
       # self.accFMU.setReal([self.VarsACC['fault_value1'].valueReference], [-1.0])
       # self.accFMU.setReal([self.VarsACC['fault_time1'].valueReference], [-1.0])
       # self.accFMU.setReal([self.VarsACC['fault_value2'].valueReference], [-1.0])
       # self.accFMU.setReal([self.VarsACC['fault_time2'].valueReference], [-1.0])

        self.time = START_TIME
        self.rows = []  # list to record the results
        self.time_trajectory = []
        self.reward_trajectory = []
        self.acceleration_trajectory = []
        self.get_rel_dis_trajectory = []
        self.get_velocity_trajectory = []
        self.safe_distance_trajectory = []
        self.get_longi_velocity_trajectory = []
        self.teller = 0
        self.teller2 = 0
        self.meanJitter = 0


        # Reset accel and relative distance
        self.observation = np.array([0, 0, 0, 0]).astype(np.float32)

        return self.observation  # reward, done, info can't be included

    def close(self):
        # close model
        self.accFMU.terminate()
        # call the FMI API directly to avoid unloading the share library
        self.accFMU.freeInstance()
        # clean up
        shutil.rmtree(self.accFMUExtractDir, ignore_errors=True)

        self.time = START_TIME
        self.rows = []  # list to record the results
        self.time_trajectory = []
        self.reward_trajectory = []
        self.acceleration_trajectory = []
        self.get_rel_dis_trajectory = []
        self.get_velocity_trajectory = []
        self.safe_distance_trajectory = []
        self.jitterArray = []
