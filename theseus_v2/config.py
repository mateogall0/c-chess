import os

DEBUG = os.getenv('TH_DEBUG', False)
ENV_ID = os.getenv('TH_ENV_ID','Chess-v0')
SYZYGY_ONLY = os.getenv('TH_SYZYGY_ONLY', False)
NO_SYZYGY = os.getenv('TH_NO_SYZYGY', False)
EXTERNAL_EVALUATION_TIME_LIMIT = float(os.getenv('TH_EXTERNAL_EVALUATION_TIME_LIMIT', 0.1))
EXTERNAL_EVALUATION_DEPTH_LIMIT = int(os.getenv('TH_EXTERNAL_EVALUATION_DEPTH_LIMIT', 20))
input_shape = 8, 8, 17
reward_factor = 1500
NUM_ENVS = int(os.getenv('TH_NUM_ENVS', 3))
alphazero_action_space = 4672