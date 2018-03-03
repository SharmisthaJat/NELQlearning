import os
from deeprl_hw2.hyperparams import EVAL_FREQ
import matplotlib.pyplot as plt
import numpy as np

import gym

# base_dir, run no, episodes_per_evaluation
datas = [
  #("atari-v0", 271, 1),
  ("psc_data", 18, 20, "DQN", "dqn"),
  #("psc_data", 17, 20),
  #("atari-v0", 251, 20),
  ("atari-v0", 274, 1, "Linear Q Network (no replay, no fixing)", "lqn_nomem")
]

env_name = "SpaceInvadersDeterministic-v3"

SHOW_ITERS = 2e6

# 271 is dueling.
# 18 is still running

def process(total_rewards, episodes_per_eval, prefix="", fname=""):
  split = [total_rewards[i : i + episodes_per_eval] for i in range(0, len(total_rewards), episodes_per_eval)]
  avgs = [sum(rewards) / episodes_per_eval for rewards in split]
  denom = episodes_per_eval - 1 if episodes_per_eval > 1 else 1
  std_devs = [np.sqrt(sum((np.array(rewards) - avg) ** 2) / denom) for rewards, avg in zip(split, avgs)]
  max_reward = max([max(rewards) for rewards in split])

  print("\t%d evaluations (max reward %0.1f)" % (len(avgs), max_reward))

  avgs = avgs[:int(SHOW_ITERS / EVAL_FREQ)]
  std_devs = std_devs[:int(SHOW_ITERS / EVAL_FREQ)]

  plt.figure()
  #plt.errorbar(EVAL_FREQ * np.array(range(1, len(avgs) + 1)), avgs, std_devs, ecolor='red', linewidth=2, elinewidth=1, fmt='o-', capsize=3)
  plt.plot(EVAL_FREQ * np.array(range(1, len(avgs) + 1)) / 1e6, avgs, '.-')
  plt.xlabel('Iteration # (millions)')
  plt.ylabel('Total Reward')
  plt.title(prefix)
  plt.xlim((0, 2))
  plt.ylim((0, 1000))
  plt.show(block=False)
  plt.savefig('../figs/reward_%s.pdf' % fname)

if __name__ == "__main__":
  for data_dir, run_no, episodes_per_eval, title, fname in datas:
    DIR = os.path.join(data_dir, "%s-run%d" % (env_name, run_no))
    print(DIR,)
    results = gym.wrappers.monitoring.load_results(DIR)
    if results is None:
      print("No results in", DIR)
      continue

    process(results['episode_rewards'], episodes_per_eval, title, fname)
            #"%s (run no. %d)" % (data_dir, run_no))

  plt.show()
