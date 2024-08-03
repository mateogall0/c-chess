#!/usr/bin/env python3

import theseus.engine as th

bot = th.Bot(new_model=False, path='./theseus/theseus.h5')

bot.play(engine_only=True)
