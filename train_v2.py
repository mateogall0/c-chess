#!/usr/bin/env python3
from theseus_v2 import theseus as engine

if __name__ == '__main__':
    """
    Used mainly for demonstration purposes
    """
    engine.train(total_timesteps=1002768)
    p = engine.auto_play(render=False)
    print(p)
    p = engine.auto_play(render=False)
    print(p)

