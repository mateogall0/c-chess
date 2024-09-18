#!/usr/bin/env python3
from theseus_v2 import theseus as engine

if __name__ == '__main__':
    """
    Used mainly for demonstration purposes
    """
    engine.train(total_timesteps=502768)
    r, p = engine.auto_play(render=False)
    print(p)
    r, p = engine.auto_play(render=False)
    print(p)

