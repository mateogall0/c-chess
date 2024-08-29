import os

DEBUG = os.getenv('TH_DEBUG', False)
ENV_ID = os.getenv('TH_ENV_ID','Chess-v0')
SYZYGY_ONLY = os.getenv('TH_SYZYGY_ONLY', False)
EXTERNAL_EVALUATION_TIME_LIMIT = float(os.getenv('TH_EXTERNAL_EVALUATION_TIME_LIMIT', 0.1))
EXTERNAL_EVALUATION_DEPTH_LIMIT = int(os.getenv('TH_EXTERNAL_EVALUATION_DEPTH_LIMIT', 20))
input_shape = (8, 8, 13)