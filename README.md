# geometry-friends-ai

## What is Geometry Friends?
Gometry Friends is a 2D platform game developed at GAIPS INESC-ID, http://gaips.inesc-id.pt/geometryfriends/.

![sample](images/human_game.gif)

This project presents this game environment for developing game-playing AI.
AI agent is trained by using reinforcement learning and inspired by the works of [yanpanlau](https://github.com/yanpanlau)
for the game of [Flappy Bird](https://github.com/yanpanlau/Keras-FlappyBird). Unlike Flappy Bird, Geoemetry Friends is a harder game to solve by pure reinforcement learning.

## Environment

Game is written for only circle agent currently. There are 9 different setups to play. These setups are under *game_setup.py* file.
There is one trained model available under */models* directory. Training is held for 2 days with TitanX GPU for almost 1 million frames.
Model specs are under *models.csv* file.

There are 3 modes available to run the game.

1. Human play mode is for you to enjoy and test the environment.
2. AI train mode is training the AI agent using reinforcement learning.
3. AI test mode is testing the AI agent.

## USAGE

## Human play
`python main.py -g 10 --setup 2`  <!-- Game over after 10 finishes of a setup -->

Parameter *-g* sets how many times the setup will be refreshed until the game is finished.
Parameter *--setup* chooses the setup (0-8).

## AI train
To start with training mode *--train* option is required. *-s* sets model saving path, *-l* model loading path (not required for new model), *--lr* learning rate and *-v* is verbosity.

<!-- Load model with -l, save with -s, --lr learning rate, -v for verbose, --setup for different setups -->
`python main.py --train -s models/<model_name>.h5 -l models/<model_name>.h5 --lr 1e-5 -g 100 -v`

`python main.py --train -s models/<model_name>.h5 --lr 1e-5 -g 100 -v`		<!-- New model, no load -->

## AI test
To start with test mode *--test* option is required. And choose setup with *--setup*.

`python main.py --test -l models/<model_name>.h5 -g 100 --setup 2 -v`
