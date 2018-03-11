import sys
sys.path.append("../nel_framework/nel/")
from config import *
import nel
import numpy as np

class Environment():
  def __init__(self,config):
    self.simulator = nel.Simulator(sim_config=config)
		self.prev_viz = np.zeros((3,3,3)) 

  def reward(self,move):
		cell = None
		if move == nel.Direction.RIGHT:
				cell = self.prev_viz[2,1,2]
		
		if move == nel.Direction.LEFT:
				cell = self.prev_viz[2,1,0]
			
		if move == nel.Direction.UP:
				cell = self.prev_viz[2,0,1]
			
		if move == nel.Direction.DOWN:
				cell = self.prev_viz[2,2,1]
		assert cell != None,'Error: Agent visual field not observed'
    return cell

  def step(self, agent):
		steps=1
		ag_move = agent.next_move()
		self.simulator.move(agent,ag_move,steps)
		curr_reward = self.reward(ag_move)
		self.prev_viz = agent.vision()
    return ag_move, curr_reward

if(__name__=='__main__'): 
  # test code
  e = Environment(config1)
  print e.reward()
