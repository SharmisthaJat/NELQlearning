# NELQlearning
nel subcommittee code 

# Getting it running
The recommended version of Python is 2.7

1. Install the nel-framework according to https://github.com/eaplatanios/nel_framework/#installation-instructions

2. Install the dependencies using `pip install requirements.txt`

3. Run train.py to train, and test.py to visualize

# Making a new agent
Create a new class that extends `BaseAgent`. For this agent, implement the following functions:
  * `save()`: Saves the state of the agent. If you don't want to save, just use a `pass` statement for the body
  * `_load()`: Loads the state of the agent. If you don't want to load, just use a `pass` statement for the body
  * `step()`: You can add whatever parameters you want to this. This will be called in your training loop. This function should call `env.step` which takes in a reference to the agent and a function that takes in no arguments and returns the next move that your agent should do.
  * `next_move()`: You can add whatever parameters you want to this. This function should return the next move that the agent should do.

Note: The recommended way you should structure the interaction between the environment and the agent is by creating a function that takes in the parameters needed for `next_move()` and returns a partial function that takes in no arguments that can be called by `env.step`. See the example in `RLAgent`.

# Visualisation

Youtube video [[paper](https://www.youtube.com/watch?time_continue=3&v=ARjEmqJNgVc)]

# Presentation

Class presentation [[link](https://docs.google.com/presentation/d/1sKSyohH7e7GUD1pb2_IIN1vnrG8tY-_Ml8qZLrlYAMk/edit?usp=sharing)]
