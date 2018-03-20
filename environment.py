import sys
sys.path.append("../nel_framework/nel/")
from config import *
import nel
import numpy as np

class Environment():
  def __init__(self, config):
    self.simulator = nel.Simulator(sim_config=config)
    self.prev_viz = np.zeros((3,3,3))   

  def step(self, agent, epsilon=0.0):
    steps=1
    ag_move = agent.next_move(epsilon)
    self.simulator.move(agent,ag_move,steps)
    curr_reward = agent.collected_items()
    #print "BEFORE"
    #print self.prev_viz
    self.prev_viz = agent.vision()
    #print "AFTER"
    #print self.prev_viz
    #print '\n'
    return ag_move, curr_reward

if(__name__=='__main__'): 
  # test code
  e = Environment(config1)
  print e.reward()
