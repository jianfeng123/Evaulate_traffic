import argparse
import configparser
import logging
import torch
from torch.autograd import Variable
import utils
import numpy as np
from tqdm import tqdm
from envs.traffic_env import large_grid
from algorithms.MACC.macc import MACC
from algorithms.IDQN.DQN import DQN
from algorithms.MAAAC.MAAAC import MAAAC
from algorithms.greedy.greedy import Greedy
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt

USE_CUDA = False  # torch.cuda.is_available()

MACC_DIR = "../results/MACC/model/incremental/max_reward_model_ep_157.pt"
DQN_DIR = "../results/DQN/model/incremental/max_reward_model_ep_65.pt"
MAAAC_DIR = "../results/M3AC/model/incremental/max_reward_model_ep_196.pt"

DQN_TRAIN_DIR = "../results/DQN/model/incremental/max_reward_model_ep_246.pt"
MACC_TRAIN_DIR = "../results/DQN/model/incremental/max_reward_model_ep_246.pt"


def plot_board(xlabe='Episodes', ylabel='Average reward'):
	plt.grid(linestyle="--", linewidth=0.3)
	plt.rcParams['font.sans-serif'] = ['Arial']  # 如果要显示中文字体，则在此处设为：SimHei
	plt.rcParams['axes.unicode_minus'] = False  # 显示负号
	ax = plt.gca()
	ax.spines['top'].set_visible(False)  # 去掉上边框
	ax.spines['right'].set_visible(False)  # 去掉右边框
	plt.xlabel(xlabe, size=16, fontweight='bold')
	plt.ylabel(ylabel, size=16, fontweight='bold')
	plt.xticks(fontsize=16, fontweight='bold')
	plt.yticks(fontsize=16, fontweight='bold')
def plot_legend():
	plt.legend(loc="best", numpoints=1)
	leg = plt.gca().get_legend()
	ltext = leg.get_texts()
	plt.setp(ltext, fontsize=16, fontweight='bold')  # 设置图例字体的大小和粗细
def plot_train_results():
	def plot_line(episode_length,data, name, color, label):
		plt.plot(range(episode_length), data[name], color=color, label=label)
	base_dir = args.base_dir
	dirs = utils.init_dir(base_dir)
	config_dir = args.config_dir
	config = configparser.ConfigParser()
	config.read(config_dir)
	episode_length_sec = config.getint('ENV_CONFIG', 'episode_length_sec')
	control_interval_sec = config.getint('ENV_CONFIG', 'control_interval_sec')
	episode_length = episode_length_sec // control_interval_sec

	dqn_results = pickle.load(open(DQN_TRAIN_DIR, 'rb'))
	macc_results = pickle.load(open(DQN_TRAIN_DIR, 'rb'))
	maa2c_name = "MAA2C"
	dqn_name = "DQN"
	# plot rewards
	plot_board('Episodes', 'Average reward')
	plot_line(episode_length, dqn_results, 'rewards', 'blue', dqn_name)
	plot_line(episode_length, macc_results, 'rewards', 'r', maa2c_name)
	plot_legend()
	plt.savefig(os.path.join(dirs['data'], 'Compare_with_train_reward' + '.png'), dpi=600)
	plt.show()

	# plot queue
	plot_board('Episodes', 'Average queue length')
	plot_line(episode_length, dqn_results, 'queues', 'blue', dqn_name)
	plot_line(episode_length, macc_results, 'queues', 'r', maa2c_name)
	plot_legend()
	plt.savefig(os.path.join(dirs['data'], 'Compare_with_train_queue' + '.png'), dpi=600)
	plt.show()

	# plot waiting
	plot_board('Episodes', 'Average waiting time')
	plot_line(episode_length, dqn_results, 'waiting', 'blue', dqn_name)
	plot_line(episode_length, macc_results, 'waiting', 'r', maa2c_name)
	plot_legend()
	plt.savefig(os.path.join(dirs['data'], 'Compare_with_train_waiting' + '.png'), dpi=600)
	plt.show()

def parse_args():
	default_base_dir = '../results/evaluate'
	default_config_dir = '../config/config_evaluate_large.ini'
	parser = argparse.ArgumentParser()
	parser.add_argument('--base-dir', type=str, required=False,
	                    default=default_base_dir, help="experiment base dir")
	parser.add_argument('--config-dir', type=str, required=False,
	                    default=default_config_dir, help="experiment config path")
	parser.add_argument('--option', type=str, required=False,
	                    default="evaluate", help="experiment config path")
	args = parser.parse_args()
	return args


def init_env(config, port=0, naive_policy=False):
	if config.get('scenario') == 'large_grid':
		return large_grid(config)


def evaluate_learning_model(env, model, episode_length):
	obs = env.start()
	waiting = []
	queues = []
	all_rewards = []
	tqdm_e = tqdm(range(episode_length), desc='reward', leave=True, unit=" episodes")
	for et_i in tqdm_e:
		torch_obs = [Variable(torch.Tensor(obs[i][np.newaxis, :]), requires_grad=False)
		             for i in range(model.nagents)]
		torch_agent_actions = model.step(torch_obs, explore=False)
		agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
		real_actions = [np.argmax(a, axis=1)[0] for a in agent_actions]
		next_obs, rewards, dones, global_reward = env.step(real_actions)  # decode one-hot
		obs = next_obs
		waiting.append(env.average_waiting_time())
		queues.append(env.curr_queue_length())
		all_rewards.append(np.mean(rewards))
	env.end()
	return waiting, queues, all_rewards


def evaluate_greedy(env, model, episode_length):
	obs = env.start(algorithm='greedy')
	tqdm_e = tqdm(range(episode_length), desc='reward', leave=True, unit=" episodes")
	waiting = []
	queues = []
	all_rewards = []
	for et_i in tqdm_e:
		actions = model.forward(obs)
		next_obs, rewards, dones, global_reward = env.step(actions, 'greedy')  # decode one-hot
		obs = next_obs
		waiting.append(env.average_waiting_time())
		queues.append(env.curr_queue_length())
		all_rewards.append(np.mean(rewards))
	env.end()
	return waiting, queues, all_rewards


def evaluate_actuated(env, episode_length):
	obs = env.start(algorithm='greedy')
	tqdm_e = tqdm(range(episode_length), desc='reward', leave=True, unit=" episodes")
	waiting = []
	queues = []
	for et_i in tqdm_e:
		env.step(actions=None, algorithm='actuated')  # decode one-hot
		waiting.append(env.average_waiting_time())
		queues.append(env.curr_queue_length())
	env.end()
	return waiting, queues


def plot_results(args):
	def plot_fill(episode_length, data, color, label, alpha):
		plt.plot(range(episode_length), data.mean(axis=0), color=color, label=label)
		plt.fill_between(range(episode_length), data.mean(axis=0) - data.std(axis=0),
						 data.mean(axis=0) + data.std(axis=0), color=color, alpha=alpha)

	base_dir = args.base_dir
	dirs = utils.init_dir(base_dir)
	config_dir = args.config_dir
	config = configparser.ConfigParser()
	config.read(config_dir)

	results = pickle.load(open(dirs['data'] + 'results.pkl', 'rb'))
	macc_waiting = np.array(results['macc_waiting'])
	macc_queues = np.array(results['macc_queues'])
	macc_rewards = np.array(results['macc_rewards'])

	mtac_waiting = np.array(results['mtac_waiting'])
	mtac_queues = np.array(results['mtac_queues'])
	mtac_rewards = np.array(results['mtac_rewards'])

	dqn_waiting = np.array(results['dqn_waiting'])
	dqn_queues = np.array(results['dqn_queues'])
	dqn_rewards = np.array(results['dqn_rewards'])

	greedy_waiting = np.array(results['greedy_waiting'])
	greedy_queues = np.array(results['greedy_queues'])
	greedy_rewards = np.array(results['greedy_rewards'])

	actuated_waiting = np.array(results['actuated_waiting'])
	actuated_queues = np.array(results['actuated_queues'])

	episode_length_sec = config.getint('ENV_CONFIG', 'episode_length_sec')
	control_interval_sec = config.getint('ENV_CONFIG', 'control_interval_sec')
	episode_length = episode_length_sec // control_interval_sec

	alpha = 0.2
	maa2c_name = "MAA2C"
	mtac_name = "mtac"
	dqn_name = "DQN"
	greedy_name = "Greedy"
	actuated_name = "Actuated"
	# plot wait
	plot_board('Intervals', "Average waiting time (s/veh)")
	plot_fill(episode_length, macc_waiting, 'r', maa2c_name, alpha)
	plot_fill(episode_length, dqn_waiting, 'y', dqn_name, alpha)
	plot_fill(episode_length, greedy_waiting, 'b', greedy_name, alpha)
	plot_fill(episode_length, actuated_waiting, 'g', mtac_name, alpha)
	plot_fill(episode_length, mtac_waiting, 'm', actuated_name, alpha)
	plot_legend()
	plt.savefig(os.path.join(dirs['data'], 'Compare_with_waiting' + '.png'), dpi=600)
	plt.show()

	# plot queue
	plot_board('Intervals', "Average queue length (veh)")
	plot_fill(episode_length, macc_queues, 'r', maa2c_name, alpha)
	plot_fill(episode_length, dqn_queues, 'y', dqn_name, alpha)
	plot_fill(episode_length, greedy_queues, 'b', greedy_name, alpha)
	plot_fill(episode_length, actuated_queues, 'g', actuated_name, alpha)
	plot_fill(episode_length, mtac_queues, 'm', mtac_name, alpha)
	plot_legend()
	plt.savefig(os.path.join(dirs['data'], 'Compare_with_queue' + '.png'), dpi=600)
	plt.show()

	# plot reward
	plot_board('Intervals', 'Average interval reward')
	plot_fill(episode_length, macc_rewards, 'r', maa2c_name, alpha)
	plot_fill(episode_length, dqn_rewards, 'y', dqn_name, alpha)
	plot_fill(episode_length, greedy_rewards, 'b', greedy_name, alpha)
	plot_fill(episode_length, mtac_rewards, 'm', mtac_name, alpha)
	plot_legend()
	plt.savefig(os.path.join(dirs['data'], 'Compare_with_reward' + '.png'), dpi=600)
	plt.show()


def evaluate(args):
	base_dir = args.base_dir
	dirs = utils.init_dir(base_dir)
	utils.init_log(dirs['log'])
	config_dir = args.config_dir
	utils.copy_file(config_dir, dirs['data'])
	config = configparser.ConfigParser()
	config.read(config_dir)

	# init centralized or multi agent
	seed = config.getint('ENV_CONFIG', 'seed')
	torch.manual_seed(seed)
	np.random.seed(seed)

	# init env
	env = init_env(config['ENV_CONFIG'])
	logging.info('Training: state space : %r, action space: %r' %
	             (env.observation_space, env.action_space))
	# init model
	macc = MACC.init_from_save(MACC_DIR)
	dqn = DQN.init_from_save(DQN_DIR)
	mtac = MAAAC.init_from_save(MAAAC_DIR)
	greedy = Greedy(env.node_names)
	macc.prep_rollouts(device='cpu')
	dqn.prep_rollouts(device='cpu')
	mtac.prep_rollouts(device='cpu')

	n_episodes = config.getint('MODEL_CONFIG', 'n_episodes')
	episode_length = env.episode_length_sec // (env.control_interval_sec)

	macc_waiting = []
	macc_queues = []
	macc_rewards = []
	mtac_waiting = []
	mtac_queues = []
	mtac_rewards = []
	dqn_waiting = []
	dqn_queues = []
	dqn_rewards = []
	greedy_waiting = []
	greedy_queues = []
	greedy_rewards = []
	actuated_waiting = []
	actuated_queues = []
	for ep_i in range(0, n_episodes):
		waiting, queues, rewards = evaluate_learning_model(env, mtac, episode_length)
		mtac_waiting.append(waiting), mtac_queues.append(queues), mtac_rewards.append(rewards)

		waiting, queues, rewards = evaluate_learning_model(env, macc, episode_length)
		macc_waiting.append(waiting), macc_queues.append(queues), macc_rewards.append(rewards)


		waiting, queues, rewards = evaluate_learning_model(env, dqn, episode_length)
		dqn_waiting.append(waiting), dqn_queues.append(queues), dqn_rewards.append(rewards)

		waiting, queues, rewards = evaluate_greedy(env, greedy, episode_length)
		greedy_waiting.append(waiting), greedy_queues.append(queues), greedy_rewards.append(rewards)

		waiting, queues = evaluate_actuated(env, episode_length)
		actuated_waiting.append(waiting), actuated_queues.append(queues)

		logging.info('train %i, macc ave waiting is %.2f, ave queue is %.2f, ave rewards is %.2f' % (
			ep_i, macc_waiting[-1][-1], np.mean(macc_queues[-1]), np.mean(macc_rewards[-1])))

		logging.info('train %i, mtac ave waiting is %.2f, ave queue is %.2f, ave rewards is %.2f' % (
			ep_i, mtac_waiting[-1][-1], np.mean(mtac_queues[-1]), np.mean(mtac_rewards[-1])))

		logging.info('train %i, dqn ave waiting is %.2f, ave queue is %.2f, ave rewards is %.2f' % (
			ep_i, dqn_waiting[-1][-1], np.mean(dqn_queues[-1]), np.mean(dqn_rewards[-1])))

		logging.info('train %i, greedy ave waiting is %.2f, ave queue is %.2f, ave rewards is %.2f' % (
			ep_i, greedy_waiting[-1][-1], np.mean(greedy_queues[-1]), np.mean(greedy_rewards[-1])))

		logging.info('train %i, actuated ave waiting is %.2f, ave queue is %.2f' % (
			ep_i, actuated_waiting[-1][-1], np.mean(actuated_queues[-1])))
	results = {}
	results['macc_waiting'] = macc_waiting
	results['macc_queues'] = macc_queues
	results['macc_rewards'] = macc_rewards

	results['mtac_waiting'] = mtac_waiting
	results['mtac_queues'] = mtac_queues
	results['mtac_rewards'] = mtac_rewards

	results['dqn_waiting'] = dqn_waiting
	results['dqn_queues'] = dqn_queues
	results['dqn_rewards'] = dqn_rewards

	results['greedy_waiting'] = greedy_waiting
	results['greedy_queues'] = greedy_queues
	results['greedy_rewards'] = greedy_rewards

	results['actuated_waiting'] = actuated_waiting
	results['actuated_queues'] = actuated_queues

	with open(dirs['data'] + 'results.pkl', "wb") as f:
		pickle.dump(results, f)


if __name__ == '__main__':
	args = parse_args()
	if args.option == 'evaluate':
		evaluate(args)
	plot_results(args)
