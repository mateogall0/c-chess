#!/usr/bin/env python3
from theseus_v2 import Engine

if __name__ == '__main__':
    """
    Used mainly for demonstration purposes
    """
    engine = Engine()
    engine.train(total_timesteps=6000)
    r, p = engine.auto_play(render=False)
    print(p)
