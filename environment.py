import sys
sys.path.append("../nel_framework/nel/")
from config import *
import nel
import numpy as np


class Environment:
  def __init__(self, config):
    self.simulator = nel.Simulator(sim_config=config)
    self.prev_viz = np.zeros((3,3,3))   
    self.prev_jelly_count = 0.0

  def step(self, agent, move_fn):
    steps = 1
    ag_move = move_fn()
    self.simulator.move(agent, ag_move, steps)
    curr_reward = int(agent.collected_items()[2]) - self.prev_jelly_count
    self.prev_jelly_count = int(agent.collected_items()[2])
    self.prev_viz = agent.vision()
    return ag_move, curr_reward


if(__name__=='__main__'): 
  # test code
  e = Environment(config1)
 