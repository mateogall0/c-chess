#!/usr/bin/env python3
import theseus as th

new = th.Theseus(new_model=True)

new.default_session_train()
new.engine_save()
print(new.engine_summary)
print(new.is_new)

#new.default_session_train()

