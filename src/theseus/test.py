#!/usr/bin/env python3
import theseus as th

new = th.Theseus(new_model=True)

history = new.session_train_model(exploration_prob=1,
                                 batch_size=512,
                                 play_iterations=10, epochs=200,
                                 exploration_prob_diff_times=5,
                                 training_iterations=10)

new.plot_training_records()
