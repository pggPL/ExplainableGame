import torch
from model import ValueFunction, board_to_input
from .env import GameEnvironment


# Function to play the game against the trained model
def play_with_model(game, model, human_player=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Continue playing until the game is over
    while not game.is_game_over():
        player = game.current_player
        # Human player's turn
        if player == human_player:
            print("Your turn!")
            game.print_board()
            move = input("Enter your move as row, col (0-indexed): ")
            row, col = map(int, move.strip().split(","))
        # Model's turn
        else:
            print("Model's turn...")
            q_values = [
                model(
                    board_to_input(
                        game.make_move(row, col, player),
                        game.player_symbols
                    ).to(device)
                )
                for row, col in game.get_valid_moves()
            ]
            max_index = q_values.index(max(q_values))
            row, col = game.get_valid_moves()[max_index]

        # Make the chosen move
        game.make_move(row, col, player)

    # Print the final game board
    game.print_board()

    # Display the game result
    if game.has_player_won(human_player):
        print("You won!")
    elif game.has_player_won(1 - human_player):
        print("Model won!")
    else:
        print("It's a draw!")


# Define the winning patterns
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

# Initialize the game environment with the winning patterns
game = GameEnvironment(winning_patterns)

# Load the trained model
model_path = "./model_files/model.pth"
model = ValueFunction()
model.load_state_dict(torch.load(model_path))

# Start playing the game with the trained model
play_with_model(game, model)
