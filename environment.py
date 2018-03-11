import sys
sys.path.append("../nel_framework/nel/")
from config import *
import nel

class Environment():
  def __init__(self,config):
    self.simulator = nel.Simulator(sim_config=config)

  def reward(self):
    return 0

  def step(self, agent):
    return 0, 0

if(__name__=='__main__'): 
  # test code
  e = Environment(config1)
  print e.reward()
