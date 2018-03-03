#Better to declare these as google flags or in JSON format
# From TAs [DO NOT CHANGE, copied from HW2 writeup]
GAMMA = 0.99
LR = 0.00025
EPSILON = 0.10
TRAIN_FREQ = 4
WINDOW_SIZE = 4
IMAGE_SIZE = 84
REPLAY_BUFFER_SIZE = 1000000
TARGET_UPDATE_FREQ = 10000
BATCH_SIZE = 32
LOG_DIR = './'

# Other Misc parameters
NUM_ITERATIONS = 20000000
NUM_ITERATIONS_LINEAR_DECAY = 1000000
CHECKPOINT_SAVE_FREQ = 100000
CHECKPOINT_MAX_TO_KEEP = 500
EVAL_FREQ = 10000 # Evaluate policy after this many network updates.
EVAL_NUM_EPISODES = 20 # Number of episodes to run per evaluation.
REPLAY_START_SIZE = 50000 # This should always be bigger than BATCH_SIZE.
MAX_REWARD_LIMIT = 1.0
MIN_REWARD_LIMIT = -1.0
REPORT_FREQ = 200
