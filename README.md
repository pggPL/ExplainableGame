# Tic-tac-toe-like Game with Neural Network

This project is an implementation of a tic-tac-toe-like game with a 10x10 board and customizable winning patterns. The game is played against a neural network model trained using the SARSA algorithm. The project consists of several Python files, each responsible for a different aspect of the game and model.

## Files

1. `game_environment.py` - Contains the `GameEnvironment` class, which defines the game environment, rules, and winning patterns.
2. `model.py` - Contains the `ValueFunction` class, which defines the neural network used as a value function for the game states.
3. `train.py` - Contains the `train` function, which trains the neural network using the SARSA algorithm.
4. `__init__.py` - A script to play the game against the trained model.
5. `README.md` - This file.

## Setup

1. Ensure you have Python 3.6 or higher installed.
2. Install PyTorch by following the instructions on the [official website](https://pytorch.org/get-started/locally/).
3. Clone this repository to your local machine.
4. Run `train.py` to train the neural network model.
5. After training, the model will be saved in the `./model_files` directory.

## Usage

1. Run `__init__.py` to play the game against the trained model.
2. During your turn, input your move as row and column indices (0-indexed) separated by a comma. For example: `3, 4`.
3. The game will alternate between the human player and the model until one player wins or the game ends in a draw.
4. The game board and the final result will be displayed on the screen.

## Customization

You can customize the winning patterns by modifying the `winning_patterns` variable in the `__init__.py` file. The game will automatically adjust to the new winning patterns.

Example of `winning_patterns`:

```python
winning_patterns = [
    [
        [1, 1],
        [1, 1]
    ],
    [
        [1, 1, 1, 1]
    ],
    [
        [1],
        [1],
        [1],
        [1]
    ]
]
