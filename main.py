from typing import Any, Dict

import gym
import numpy as np
import optuna
import torch

from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from stable_baselines3 import DDPG
from stable_baselines3 import PPO
from stable_baselines3 import TD3
from stable_baselines3 import SAC


# from stable_baselines.common.env_util import make_vec_env
# from stable_baselines.common.callbacks import BaseCallback
from scipy.io import savemat
from torch import nn

import RL
from RL import FMIEnv

N_TRIALS = 20
N_STARTUP_TRIALS = 5
N_EVALUATIONS = 2
N_TIMESTEPS = 10000
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
N_EVAL_EPISODES = 3
TIMESTEPS = 10000
DEFAULT_HYPERPARAMS = {
    "policy": "MlpPolicy",
    "env": FMIEnv,
}

def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for PPO hyperparams.
    :param trial:
    :return:
    """

    gamma = trial.suggest_categorical("gamma", [0.8, 0.88, 0.98, 0.99, 0.995])
    vf_coef = trial.suggest_uniform("vf_coef", 0.5, 1)
    n_epochs = trial.suggest_categorical("n_epochs", [3, 5, 10, 20, 30])
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 256, 512, 1024,2048])
    n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.90, 0.95, 0.98, 0.99, 1.0])
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1)
    ent_coef = trial.suggest_loguniform("ent_coef", 0.00000001, 0.1)
    return {

        "gamma": gamma,
        "vf_coef": vf_coef,
        "n_epochs":n_epochs,
        "batch_size": batch_size,
        "gae_lambda":gae_lambda,
        "clip_range":clip_range,
        "n_steps": n_steps,
        "learning_rate": learning_rate,
        "ent_coef":ent_coef,

    }

def sample_td3_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for TD3 hyperparams.
    :param trial:
    :return:
    """

    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1)
    batch_size = trial.suggest_categorical("batch_size", [100, 128, 256, 512, 1024, 2048])
    buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(1e5), int(1e6)])
    # Polyak coeff
    tau = trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.02, 0.05, 0.08])
    target_policy_noise = trial.suggest_categorical("target_policy_noise", [0.05, 0.1, 0.15, 0.2, 0.25])

    return {

        "gamma": gamma,
        "learning_rate":learning_rate,
        "batch_size":batch_size,
        "buffer_size":buffer_size,
        "tau":tau,
        "target_policy_noise": target_policy_noise

    }

def sample_ddpg_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for TD3 hyperparams.
    :param trial:
    :return:
    """


    gamma = trial.suggest_categorical("gamma", [0.8, 0.95, 0.98, 0.99, 0.995, 0.999,])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 100, 128, 256, 512, 1024, 2048])
    buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(1e5), int(1e6)])
    # Polyak coeff
    tau = trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.02, 0.05, 0.08])





    return {

        "gamma": gamma,
        "tau": tau,
        "learning_rate":learning_rate,
        "batch_size": batch_size,
        "buffer_size":buffer_size,


    }

def sample_sac_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for TD3 hyperparams.
    :param trial:
    :return:
    """

    gamma = trial.suggest_categorical("gamma", [0.87, 0.9, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256, 512, 1024, 2048])
    buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(1e5), int(1e6)])
    # train_freq = trial.suggest_categorical('train_freq', [1, 10, 100, 300])
    # Polyak coeff
    tau = trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.02, 0.05, 0.08])
   # ent_coef = trial.suggest_categorical('ent_coef', ['auto', 0.5, 0.1, 0.05, 0.01, 0.0001])



    return {

        "gamma": gamma,
        "tau": tau,
        "learning_rate":learning_rate,
        "tau":tau,
        "batch_size": batch_size,
        "buffer_size":buffer_size,



    }


class TrialEvalCallback(EvalCallback):
    """Callback used for evaluating and reporting a trial."""

    def __init__(
        self,
        eval_env: FMIEnv,
        trial: optuna.Trial,
        n_eval_episodes: int = N_EVAL_EPISODES,
        eval_freq: int = EVAL_FREQ,
        deterministic: bool = True,
        verbose: int = 1,
    ):

        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True

class Objective(object):
    def __init__(self, algo):
        self.algo = algo


    def __call__(self,trial) -> float:
    # def objective(trial:optuna.Trial) -> float:
        # reward="basic_reward"
        # algo="A2C"
        kwargs = DEFAULT_HYPERPARAMS.copy()
        env = FMIEnv()
        n_actions = env.action_space.shape[-1]
        # instead of 0.1 you can use any [0 1] + OrnsteinUhlenbeckActionNoise can be used instead of NormalActionNoise

        kwargs.update({"env": env})
        # Sample hyperparameters
        if self.algo=="TD3":
            kwargs.update(sample_td3_params(trial))
            # model = PPO("MlpPolicy", FMIEnv(reward), verbose=1)
            model = TD3(**kwargs)
        elif self.algo == "PPO":
            kwargs.update(sample_ppo_params(trial))
            # model = PPO("MlpPolicy", FMIEnv(reward), verbose=1)
            model = PPO(**kwargs)
        elif self.algo == "DDPG":
            kwargs.update(sample_ddpg_params(trial))
            model = DDPG(**kwargs)
        elif self.algo == "SAC":
            kwargs.update(sample_sac_params(trial))
            model = SAC(**kwargs)
        else:
            model =None
            print ("unvalid model for HPT")
        # Create env used for evaluation
        eval_env = FMIEnv()
        # Create the callback that will periodically evaluate
        # and report the performance
        # deterministic evaluation have better performance
        eval_callback = TrialEvalCallback(eval_env, trial, n_eval_episodes=N_EVAL_EPISODES, eval_freq=EVAL_FREQ,deterministic=True) #, deterministic=False) can be set for SAC and maybe PPO, A2C

        nan_encountered = False
        try:
            model.learn(N_TIMESTEPS, callback=eval_callback)
        except (AssertionError, ValueError) as e:
            # Sometimes, random hyperparams can generate NaN
            print(e)
            nan_encountered = True
            raise optuna.exceptions.TrialPruned()
        finally:
            # Free memory
            eval_env.reset()
            model.env.close()
            eval_env.close()

        del model.env, eval_env
        del model

        # Tell the optimizer that the trial failed
        if nan_encountered:
            return float("nan")

        if eval_callback.is_pruned:
            raise optuna.exceptions.TrialPruned()

        return eval_callback.last_mean_reward

def HPExtractor(algo):

    # Set pytorch num threads to 1 for faster training
    # torch.get_num_threads()
    torch.set_num_threads(15)

    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    # Do not prune before 1/3 of the max budget is used
    pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3)

    # todo: choose proper study name
    # Note: storage="sqlite:///test_study.db" should e removed after each HPT
    study = optuna.create_study(storage="sqlite:///ACC_"+algo+".db", sampler=sampler, pruner=pruner,
                                study_name="ACC_"+algo, direction="maximize", load_if_exists=True)

    try:
        study.optimize(Objective(algo=algo), n_trials=N_TRIALS)

    except KeyboardInterrupt:
        pass

    pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))


    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial
    best_parm = study.best_params


    print("  Value: ", trial.value)

    study.trials_dataframe().to_csv("ACC_"+algo+"_study.csv")
    print(
        "-------------------------------------\n----------------------------------------------\n----------------------------------------------\n---------------------------\n")
    print(best_parm)


    return best_parm

def main():
    # Logging bash command bellow
    # tensorboard --logdir ./a2c_cartpole_tensorboard/
    # tensorboard dev upload --logdir \'./a2c_cartpole_tensorboard/'
    env = FMIEnv()
    TIMESTEPS = N_TIMESTEPS  # env1 = make_vec_env("CartPole-v1", n_envs=4)
    useHyperParameters = 0
    makeNewModel = 0
    agentUsed = "td3"

    #Extract all hyperparameters so we can input it into our model
    if(useHyperParameters == 1 ):
        if(agentUsed=="ppo"):
            best_params = HPExtractor("PPO")
            gamma1 = best_params["gamma"]
            vf_coef1 = best_params["vf_coef"]
            n_epochs1 = best_params["n_epochs"]
            batch_size1 = best_params["batch_size"]
            gae_lambda1 = best_params["gae_lambda"]
            clip_range1 = best_params["clip_range"]
            n_steps1 = best_params["n_steps"]
            learning_rate1 = best_params["learning_rate"]
            ent_coef1 = best_params["ent_coef"]
            with open('bestparam.txt', 'w') as f:
             f.write(str(best_params))

        elif(agentUsed=="ddpg"):
            best_params = HPExtractor("DDPG")
            gamma1 = best_params["gamma"]
            tau1 =best_params["tau"]
            learning_rate1=best_params["learning_rate"]
            batch_size1=best_params["batch_size"]
            buffer_size1=best_params["buffer_size"]
            with open('bestparam.txt', 'w') as f:
                f.write(str(best_params))

        elif (agentUsed == "td3"):
            best_params = HPExtractor("TD3")
            gamma1 = best_params["gamma"]
            tau1 = best_params["tau"]
            learning_rate1 = best_params["learning_rate"]
            batch_size1 = best_params["batch_size"]
            buffer_size1 = best_params["buffer_size"]
            target_policy_noise1 = best_params["target_policy_noise"]
            with open('bestparam.txt', 'w') as f:
                f.write(str(best_params))

        elif(agentUsed=="sac"):
            best_params = HPExtractor("SAC")

            gamma1 = best_params["gamma"]
            learning_rate1 = best_params["learning_rate"]
            batch_size1 = best_params["batch_size"]
            buffer_size1 = best_params["buffer_size"]
            tau1 = best_params["tau"]
            ent_coef1 = best_params["ent_coef"]
            target_entropy1 = best_params["target_entropy"]




    if(makeNewModel==1):
        if(useHyperParameters == 1):
            if(agentUsed == "ppo"):
                model = PPO("MlpPolicy", env, gamma = gamma1, n_steps=n_steps1, learning_rate = learning_rate1, ent_coef=ent_coef1,  vf_coef = vf_coef1, clip_range= clip_range1, n_epochs=n_epochs1, batch_size=batch_size1, gae_lambda= gae_lambda1, verbose=1, tensorboard_log="./finalTests/")
            elif(agentUsed == "ddpg"):
                model = DDPG("MlpPolicy", env, gamma= gamma1 , tau = tau1, learning_rate=learning_rate1, batch_size=batch_size1, buffer_size=buffer_size1 ,  verbose=1, tensorboard_log="./finalTests/")
            elif (agentUsed == "td3"):
                model = TD3("MlpPolicy", env, gamma=gamma1, tau=tau1, learning_rate=learning_rate1,batch_size=batch_size1, buffer_size=buffer_size1, verbose=1, target_policy_noise = target_policy_noise1 , tensorboard_log="./finalTests/")
            elif (agentUsed == "sac"):
                model = SAC("MlpPolicy", env, gamma=gamma1, tau=tau1, learning_rate=learning_rate1, batch_size=batch_size1, buffer_size=buffer_size1, verbose=1, device= "auto" ,tensorboard_log="./finalTests/")
        else:
            # Always change manually to TD3 PPO SAC A2C
            model = TD3("MlpPolicy", env, verbose=1, tensorboard_log="./finalTests/")
        model.learn(total_timesteps=TIMESTEPS, tb_log_name="TD3_R1_HP")
        model.save("ACCBenchmark_Accelaration")
        del model  # remove to demonstrate saving and loading

    #Always change manually to TD3 PPO SAC A2C
    model = TD3.load("TD3_R2")
    obs = env.reset()

    # study.best_params
    for i in range(int(TIMESTEPS)):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        acceleration_array = np.array([range(0, len(env.acceleration_trajectory))]).T
        acceleration_array = acceleration_array / 1000
        acceleration_array = np.append(acceleration_array, np.array([env.acceleration_trajectory]).T, axis=1)
        savemat("ACCBenchmark_matfiles/ACC.mat", {"matrix": acceleration_array})

    env.close()


#     export to .mat file for simulink


if __name__ == "__main__":
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

