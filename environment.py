import sys
sys.path.append("../nel_framework/nel/")
from config import *
import nel
import numpy as np

class Environment():
  def __init__(self, config):
    self.simulator = nel.Simulator(sim_config=config)
    self.prev_viz = np.zeros((3,3,3)) 

  def reward(self, move):
    cell = None
    if move == nel.Direction.RIGHT:
      cell = self.prev_viz[1,2,2]

    if move == nel.Direction.LEFT:
      cell = self.prev_viz[1,0,2]

    if move == nel.Direction.UP:
      cell = self.prev_viz[0,1,2]

    if move == nel.Direction.DOWN:
      cell = self.prev_viz[2,1,2]
    assert cell != None,'Error: Agent visual field not observed'
    return cell

  def step(self, agent, epsilon=0.0):
    steps=1
    ag_move = agent.next_move(epsilon)
    self.simulator.move(agent,ag_move,steps)
    curr_reward = self.reward(ag_move)
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
