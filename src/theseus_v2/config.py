import os

DEBUG = os.getenv('THESEUS_DEBUG', False)
ENV_ID = os.getenv('ENV_ID','Chess-v0')
NUM_ENVS = os.getenv('NUM_ENVS', 1)
