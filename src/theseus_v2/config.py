import os

DEBUG = os.getenv('TH_DEBUG', False)
ENV_ID = os.getenv('TH_ENV_ID','Chess-v0')
SYZYGY_ONLY = os.getenv('TH_SYZYGY_ONLY', False)
