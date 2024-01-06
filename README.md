# Theseus - Chess Bot
Theseus is a Machine Learning algorithm coded using the Keras algorithm to perform Chess moves per given position.

This project was made as the final MVP for Holberton School's Machine Learning SPE.

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1869px-Python-logo-notext.svg.png" height=80/>
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Tensorflow_logo.svg/1915px-Tensorflow_logo.svg.png" height=80/>
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/ae/Keras_logo.svg/2048px-Keras_logo.svg.png" height=80/>
  <img src="https://cdn.worldvectorlogo.com/logos/numpy-1.svg" height=80/>
</p>

## Training sessions
This bot uses a combination of supervised learning and reinforcement learning for its training routines taking into consideration the fact that it must learn from its right guesses while also attempting to make correct predictions over data that works as a sort of puzzle for the bot. The idea is to mimic the learning that humans go through while learning chess, the way that we normally play is with a lot of practice while also consulting puzzles to get some more objectively better ways of playing.

### Reinforcement learning
The training sessions for this model consist of the bot firstly playing a given number of Chess games, beggining with a high level of randomness to the moves it makes and the bot learns from these games. Then the algorithm performs a filter for moves where the bot should not be learning from, for instance: the moves of the losing color, from games where a draw occured, and some other cases. The idea behind this is to optimize the learning routines for the bot while also filtering data that may turn out to be bad.

### Supervised learning
After this collection of data, the model uses the <code>fit</code> Keras method to learn from this data. To improve the way Theseus plays, another level of complexity must be added by using validation data. Validation data will help on the preocess of learning to get better and more precise predictions using unseen data. The objective is to get Chess positions with its best solution, fortunatelly a small part of all the Chess possibilities have already been solved and can be reached in chess positions with seven or less pieces. In this case I obtained this data from the <a href="https://syzygy-tables.info/">Syzygy tablebase</a>.

## Dependencies installation 
Dependencies for Theseus are managed using the Anaconda environments. To install Conda on your device you can follow this link: <a href="https://www.anaconda.com/download">Conda installation guide</a>.

Executing the following command on the root of the project will automatically generate a Conda environment with all the dependencies needed:
```bash
$ conda env create -f env.yml
```

After that, you can activate this environment using the following command:
```bash
$ conda activate theseus_env
```

To exit this environment you can use the following command:
```bash
(theseus_env) $ conda deactivate
```

If you want to remove this environment from your system you can use the following command:
```bash
$ conda env remove --name theseus_env
```

<img src="https://uploads-ssl.webflow.com/6105315644a26f77912a1ada/63eea844ae4e3022154e2878_Holberton.png"  height=40/>
