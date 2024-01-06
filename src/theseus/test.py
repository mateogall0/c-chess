#!/usr/bin/env python3
import theseus as th

new = th.Theseus(new_model=True)

history = new.default_session_train()
new.engine_save()
print(history)
print(new.is_new)
