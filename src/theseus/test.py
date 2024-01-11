#!/usr/bin/env python3
import theseus as th

new = th.Bot(new_model=True)

new.default_session_train()

new.engine_save()
new.plot_training_records()

