import torch
import torch.optim as optim
import random
from model import ValueFunction, board_to_input
from .env import GameEnvironment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

# Implementation of the epsilon-greedy policy
def epsilon_greedy_policy(game, value_function, state, epsilon):
    # Choose a random action with probability epsilon
    if random.random() < epsilon:
        return random.choice(game.get_valid_moves())
    # Otherwise, choose the action with the highest state-action value
    else:
        q_values = [value_function(board_to_input(game.make_move(row, col, player), game.player_symbols).to(device)) for row, col in game.get_valid_moves()]
        max_index = q_values.index(max(q_values))
        return game.get_valid_moves()[max_index]


# Function to train the neural network using the SARSA algorithm
def train(game, value_function, optimizer, num_episodes, alpha=0.1, gamma=0.99, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.999):
    # Loop through all episodes
    for episode in range(num_episodes):
        # Initialize the game
        game.__init__(winning_patterns)
        state = game.get_board_representation()
        # Apply the epsilon-greedy strategy
        epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** episode))
        player = 0
        done = False

        # While the game is not over
        while not done:
            # Choose an action according to the epsilon-greedy policy
            action = epsilon_greedy_policy(game, value_function, state, epsilon)
            # Perform the action in the game environment
            game.make_move(*action, player)
            reward = 0

            # Check if the player has won
            if game.has_player_won(player):
                next_state = None
                reward = 1
                done = True
            # If there are no more possible moves, end the game
            elif len(game.get_valid_moves()) == 0:
                next_state = None
                done = True
            # Otherwise, update the state
            else:
                next_state = game.get_board_representation()
                player = 1 - player

            # Calculate the state-action value for the current state
            current_q = value_function(board_to_input(state, game.player_symbols).to(device))
            # Calculate the state-action value for the next state
            if next_state is not None:
                next_action = epsilon_greedy_policy(game, value_function, next_state, epsilon)
                next_q = value_function(board_to_input(game.make_move(*next_action, player), game.player_symbols).to(device))
            else:
                next_q = 0

            # Update the state-action value function
            target = reward + gamma * next_q
            loss = torch.nn.MSELoss(current_q, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Move to the next state
            state = next_state

        # Display progress
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes} completed.")

# Save the trained model to the "./model_files" directory
model_save_path = "./model_files/model.pth"

value_function = ValueFunction().to(device)
torch.save(value_function.state_dict(), model_save_path)