from flask import Flask, render_template, request, jsonify
from model import ValueFunction, board_to_input
from env import GameEnvironment, winning_patterns, GameState
import torch

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
value_function = ValueFunction().to(device)
value_function.load_state_dict(torch.load("./model_files/value_function.pth"))

game_environment = GameEnvironment(winning_patterns)


@app.route('/')
def index():
    return render_template('index.html')


@app.route("/get_move", methods=["POST"])
def get_move():
    """
    Get the best move for the current state of the board.

    Returns:
        JSON response containing the best move (row, col) for the current board state.
    """
    data = request.json
    board = data["board"]
    player = data["player"]
    state = GameState(board, player)
    if game_environment.has_player_won(state, player):
        result = "win"
        return jsonify({"row": None, "col": None, "result": result, "value": None})

    q_values = [
        value_function(
            board_to_input(game_environment.make_move(state, row, col).board, game_environment.player_symbols).to(device)
        )
        for row, col in game_environment.get_valid_moves(state)
    ]
    max_index = q_values.index(max(q_values))
    best_move = game_environment.get_valid_moves(state)[max_index]
    result = None
    if game_environment.has_player_won(state, 1 - player):
        result = "lose"
    elif len(game_environment.get_valid_moves(state)) == 0:
        result = "draw"

    return jsonify({"row": best_move[0], "col": best_move[1], "result": result, "value": round(max(q_values).item(), 2)})


@app.route("/get_predictions", methods=["POST"])
def get_predictions():
    """
    Get the Q-values for all possible moves for the current state of the board.

    Returns:
        JSON response containing the Q-values for all possible moves for the current board state.
    """
    data = request.json
    board = data["board"]
    player = data["player"]
    state = GameState(board, player)
    q_values = [
        value_function(
            board_to_input(game_environment.make_move(state, row, col).board, game_environment.player_symbols).to(device)
        )
        for row, col in game_environment.get_valid_moves(state)
    ]
    return jsonify({"q_values": [round(q.item(), 2) for q in q_values]})


if __name__ == "__main__":
    app.run()
