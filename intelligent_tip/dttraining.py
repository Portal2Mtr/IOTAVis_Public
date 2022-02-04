"""
Main training for adaptive tip selection with incremental decision tree

Decision tree trained for tangle generation using conservative q-improvement first,
criterion for nodes updated after first round of training

"""

import sys
sys.path.append('./tangle_env/envs/')
sys.path.append('./')
import time
import yaml
import logging
import matplotlib.pyplot as plt
from itertools import count
from intelligent_tip._dtclass import ConsDTLearner
from tangle_env.envs.tangleEnv import TangleEnv
import numpy as np
from statistics import mean
import optuna
import sys
import argparse

logger = logging.getLogger(__name__)


class treeTrainer(object):

    def __init__(self, config):

        self.config = config

    def initTrial(self,trial):

        #Init optuna parameters
        ops = self.config['optuna']
        self.config['simParams']['learnAlpha'] = \
            trial.suggest_float("learnalpha", ops['alpha'][0],
                                ops['alpha'][1],step=ops['alpha'][2])

        self.config['simParams']['learnDecay'] = \
            trial.suggest_float("alphadecay",ops['alphadecay'][0],
                                ops['alphadecay'][1],step=ops['alphadecay'][2])

        self.config['simParams']['splitDecay'] = \
            trial.suggest_float("splitDecay", ops['splitdecay'][0],
                                ops['splitdecay'][1],step=ops['splitdecay'][2])

        self.config['simParams']['visitDecay'] = \
            trial.suggest_float("visitdecay", ops['visitdecay'][0],
                                ops['visitdecay'][1],step=ops['visitdecay'][2])

        self.config['simParams']['gamma'] = \
            trial.suggest_float("gamma", ops['gamma'][0],
                                ops['gamma'][1],step=ops['gamma'][2])

        self.config['simParams']['maxTreeDepth'] = \
            trial.suggest_int("maxtreedepth",ops['maxdepth'][0],
                              ops['maxdepth'][1],
                              ops['maxdepth'][2])


        self.config['simParams']['trimVisitPerc'] = \
            trial.suggest_float("trimvisit",
                                ops['trimvisit'][0],
                                ops['trimvisit'][1],
                                step=ops['trimvisit'][2])

        self.config['simParams']['maxQVals'] = \
            trial.suggest_int("maxqvals",
                              ops['maxq'][0],
                              ops['maxq'][1],
                              ops['maxq'][2])

        self.config['simParams']['banditbatch'] = \
            trial.suggest_int("banditbatch",
                              ops['banditbatch'][0],
                              ops['banditbatch'][1],
                              ops['banditbatch'][2])

        return trial

    def __call__(self, trial=None):
        """
        Main training loop for the stable decision tree with
        conservative q-improvement and differential decision
        trees
        :return:
        """
        # Create Tangle environment API

        config_file_data = self.config
        np.random.seed(0)

        if trial is not None:
            # Initialize selective parameters for optuna
            trial = self.initTrial(trial)

        sim_conf = config_file_data['simParams']
        env_conf = config_file_data['envParams']

        num_episodes = sim_conf['numEpisodes']
        lambda_range = sim_conf['lambdaRange']

        init_lambda = np.random.uniform(lambda_range[0], lambda_range[1])

        env_conf['lambdaVal'] = init_lambda

        env = TangleEnv(env_conf).unwrapped
        env.lambda_rate = init_lambda
        tip_logs = []
        reward_lists = []
        start_time = time.time()
        step_time = start_time
        learner_tree = ConsDTLearner(env.render(), sim_conf)
        for i_episode in range(num_episodes):
            np.random.seed(i_episode)
            # Initialize the environment and state
            env.reset()
            reward_list = []
            current_state = env.render()

            # Take actions until sim is complete
            learner_tree.learn_alpha = sim_conf['learnAlpha']
            for t in count():
                # Select and perform an action
                leaf_state = learner_tree.get_leaf(current_state)
                action = learner_tree.take_action(leaf_state)
                _, reward, done, _ = env.step(action)
                reward_list.append(reward)
                if done:
                    break

                # Observe new state
                last_state = current_state
                current_state = env.render()
                # Update q values and history
                q_change = learner_tree.update_q_values(current_state, reward, action)
                if leaf_state != 0:
                    learner_tree.update_bandit(leaf_state, action)
                learner_tree.update_crits(last_state)
                learner_tree.update_visit_freq(leaf_state)
                learner_tree.update_poss_splits(leaf_state, last_state, action, q_change)
                best_split, best_val = learner_tree.best_split(leaf_state, action)
                learner_tree.check_perform_split(leaf_state, best_split, best_val)
                if t % 100 == 0:
                    print('{}-th step... (Time: {}, Reward={}, stddev={})'.format(t, time.time() - step_time, reward,
                                                                                  current_state[-1]))
                    step_time = time.time()
                # Decay learning rate for generating tree throughout training
                learner_tree.learn_alpha = learner_tree.learn_alpha * (1 / (1 + sim_conf['learnDecay'] * t))

            # Done with episode, log

            print('# of tree nodes={}, Tree depth={}'.format(len(learner_tree.tree.nodes),
                                                             learner_tree.get_max_depth()))
            print('{}-th Episode, Reward: {},Lambda: {}, Time: {}'.format(i_episode + 1,
                                                                          mean(reward_list),
                                                                          env.lambda_rate,
                                                                          time.time() - start_time))
            start_time = time.time()
            report = env.training_report()
            print("URTS: {}, MCMC: {},tipsLeft={}".format(report['numCalcs']['urts'],
                                                          report['numCalcs']['wrw'], report['tipsLeftBehind']))

            reward_lists.append(reward_list)
            tip_logs.append(env.training_report()['tipsLog'])
            learner_tree.prune_tree()
            episode_lambda = np.random.uniform(lambda_range[0], lambda_range[1])
            env.lambda_rate = episode_lambda
            learner_tree.reset_bandit()

        print('Basic Training complete!')
        print('# of tree nodes={}, Tree depth={}'.format(len(learner_tree.tree.nodes), learner_tree.get_max_depth()))

        logger.info("Running Testing...")
        if trial is not None:
            np.random.seed(0) # Fixed single test
            # Run test with lambda = 10
            # Initialize the environment and state
            env.lambda_rate = 10
            logger.info("Testing...")
            # Initialize the environment and state
            env.reset()
            current_state = env.render()

            # Take actions until sim is complete
            learner_tree.learn_alpha = sim_conf['learnAlpha']
            for t in count():
                # Select and perform an action
                leaf_state = learner_tree.get_leaf(current_state)
                action = learner_tree.take_action(leaf_state)
                _, reward, done, _ = env.step(action)
                if done:
                    break

                # Observe new state
                current_state = env.render()
                if t % 100 == 0:
                    print('{}-th step... (Time: {}, Reward={}, stddev={})'.format(t, time.time() - step_time, reward,
                                                                                  current_state[-1]))

            report = env.training_report()
            print("Training results: URTS: {}, MCMC: {},tipsLeft={}".format(report['numCalcs']['urts'],
                                                          report['numCalcs']['wrw'], report['tipsLeftBehind']))

            # learner_tree.save_tree(trial.number)
            return report['numCalcs']['urts'], report['tipsLeftBehind']
        else:
            learner_tree.save_tree()
            return tip_logs, reward_lists


# Plots training total tips and rewards
def plot_performance(tip_logs, plot_rewards):
    """
    Plot performance of decision tree training
    :param tip_logs: Log of total number of tips over time
    :param plot_rewards: Log of total rewards over time for training
    :return: None
    """

    fig1, ax = plt.subplots(1, 1)
    x_array = [i for i in range(len(tip_logs[0]))]
    ax.plot(x_array, tip_logs[0], label='First Episode')
    ax.plot(x_array, tip_logs[-1], label='Last Episode')
    ax.set_title('Tip Calculation Results')
    ax.set_xlabel('Calculation Sample')
    ax.set_ylabel('Number of Tips')
    ax.legend()
    plt.savefig('./intelligent_tip/output/BiasedTStiplogs.png')

    fig2, ax2 = plt.subplots(1, 1)
    x_array = [i for i in range(len(plot_rewards[0]))]
    ax2.plot(x_array, plot_rewards[0], label='First Episode')
    ax2.plot(x_array, plot_rewards[-1], label='Last Episode')
    ax2.set_title('Rewards Comparison')
    ax2.set_xlabel('Calculation Sample')
    ax2.set_ylabel('Rewards')
    ax2.legend()
    plt.savefig('./intelligent_tip/output/BiasedTSrewards.png')
    plt.show()


if __name__ == "__main__":
    # Initialize seed for repeatability
    # Load the test config file
    np.random.seed(0)
    with open("./pyserver/configs/dtconfig.yaml", "r") as read_file:
        config = yaml.load(read_file, Loader=yaml.FullLoader)

    conductStudy = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--runStudy', dest='doStudy', action='store_true')
    parser.add_argument('--noStudy', dest='doStudy', action='store_false')
    parser.set_defaults(doStudy=True)
    args = parser.parse_args()
    trainer = treeTrainer(config)

    config['optuna']['seed'] = args.seed

    if args.doStudy:
        # Parameter study with optuna (Create postgresql server with parallel scripts running in tmux)
        print("Starting study!")

        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

        trainer = treeTrainer(config)
        optunaconfig = config['optuna']
        sampler = optuna.samplers.TPESampler(multivariate=True, seed=config['optuna']['seed'])
        study = optuna.study.create_study(study_name="dtstudy",
                                          directions=["minimize","minimize"],
                                          storage="postgresql://postgres@localhost",
                                          load_if_exists=True,
                                          sampler=sampler)
        # study = pickle.load(open('./intelligent_tip/output/study/banditstudy.p', 'rb'))
        # study = optuna.create_study(study_name="dtstudy", sampler=sampler,directions=["minimize","minimize"])

        study.sampler = sampler

        # Conduct Optuna
        study.optimize(trainer, n_trials=optunaconfig['num_tests'])
        # pickle.dump(study, open('./intelligent_tip/output/study/banditstudy.p', 'wb'))

        print("Best Trials:")
        for idx, trial in enumerate(study.best_trials):
            print("For Trial {}... Values: {}, Params: {}".format(trial.number, trial.values, trial.params))

    else:

        tipLogs, rewards = trainer(trial=None)
        plot_performance(tipLogs, rewards)
