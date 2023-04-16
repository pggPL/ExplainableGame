# Tic-Tac-Toe with Custom Winning Patterns

This project is an implementation of a tic-tac-toe game on a 10x10 board with custom winning patterns. The game can be played in the terminal or in a web browser. A neural network is trained using the SARSA algorithm to play the game.

## Getting Started

1. Install Python 3.7 or higher.
2. Install the required Python packages: `pip install -r requirements.txt`
3. Make `run.sh` executable: `chmod +x run.sh`

## Usage

The game can be played in the terminal or in a web browser.

### Training the Model

To train the model, run the following command:

```./run.sh train```


The trained model will be saved to `./model_files`.

### Playing the Game in the Terminal

To play the game in the terminal against the trained model, run the following command:


```./run.sh play```


### Playing the Game in a Web Browser

To play the game in a web browser, follow these steps:

1. Start the web server by running the following command:

```./run.sh server```

2. Open a web browser and navigate to `http://localhost:8080`.

## Customization

You can customize the winning patterns by modifying the `winning_patterns` variable in `train.py`. The patterns are represented as 2D lists of 1s and 0s.

## Project Structure

- `game_environment.py`: Contains the game environment and the main logic of the game.
- `model.py`: Contains the PyTorch model for the Value Function.
- `train.py`: Contains the code for training the model using the SARSA algorithm.
- `play.py`: Contains the code for playing the game in the terminal against the trained model.
- `server.py`: Contains the code for running the web server.
- `static/`: Contains the static files (HTML, CSS, and JavaScript) for playing the game in a web browser.
- `requirements.txt`: Lists the required Python packages.
- `run.sh`: A shell script for running different parts of the project.
- `README.md`: This file.
