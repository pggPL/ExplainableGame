import torch
from model import ValueFunction, board_to_input
from env import GameEnvironment, winning_patterns, GameState


# Function to play the game against the trained model
def play_with_model(game_environment: GameEnvironment, model: ValueFunction, human_player=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    state: GameState = game_environment.initial_state()

    # Continue playing until the game is over
    while not game_environment.is_terminal(state):
        player = state.current_player
        # Human player's turn
        if player == human_player:
            print("Your turn!")
            game_environment.print_board(state)
            move = input("Enter your move as row, col (0-indexed): ")
            row, col = map(int, move.strip().split(","))
        # Model's turn
        else:
            print("Model's turn...")
            q_values = [
                model(
                    board_to_input(
                        game_environment.make_move(state, row, col),
                        game_environment.player_symbols
                    ).to(device)
                )
                for row, col in game_environment.get_valid_moves(state)
            ]
            max_index = q_values.index(max(q_values))
            row, col = game_environment.get_valid_moves(state)[max_index]

        # Make the chosen move
        state = game_environment.make_move(state, row, col)

    # Print the final game board
    game_environment.print_board(state)

    # Display the game result
    if game_environment.has_player_won(state, human_player):
        print("You won!")
    elif game_environment.has_player_won(state, 1 - human_player):
        print("Model won!")
    else:
        print("It's a draw!")


# Initialize the game environment with the winning patterns
env = GameEnvironment(winning_patterns)

# Load the trained model
model_path = "./model_files/model.pth"
model = ValueFunction()
model.load_state_dict(torch.load(model_path))

# Start playing the game with the trained model
play_with_model(env, model)
