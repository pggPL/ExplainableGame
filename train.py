import torch
import torch.nn.functional as F
import torch.optim as optim
import random
from typing import Tuple
from model import ValueFunction, board_to_input
from env import GameEnvironment, GameState, winning_patterns


# Define the epsilon-greedy policy for selecting actions during training
def epsilon_greedy_policy(
        game_environment: GameEnvironment,
        value_function: ValueFunction,
        state: GameState,
        epsilon: float,
        is_target: bool = False) -> Tuple[int, int]:
    if random.random() < epsilon or is_target:
        return random.choice(game_environment.get_valid_moves(state))
    else:
        q_values = [
            value_function(
                board_to_input(game_environment.make_move(state, row, col).board,
                               game_environment.player_symbols).to(device)
            )
            for row, col in game_environment.get_valid_moves(state)
        ]
        max_index = q_values.index(max(q_values))
        return game_environment.get_valid_moves(state)[max_index]

# Define the training loop using the DQN algorithm
def train(
        game: GameEnvironment,
        value_function: ValueFunction,
        target_value_function: ValueFunction,
        optimizer: optim.Optimizer,
        num_episodes: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay: float = 0.999,
        target_update_frequency: int = 100) -> None:
    for episode in range(num_episodes):
        # Initialize the state for the current episode
        state = game.initial_state()
        # Update the epsilon value for exploration
        epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** episode))

        l = 0
        # Continue until the game reaches a terminal state
        while not game.is_terminal(state):
            l += 1
            # Choose an action based on the epsilon-greedy policy
            action = epsilon_greedy_policy(game, value_function, state, epsilon)
            # Get the next state after performing the chosen action
            next_state = game.make_move(state, *action)

            # Update the state for the next iteration
            state = next_state

            # If the game is not over, the target network will play
            if not game.is_terminal(state):
                target_action = epsilon_greedy_policy(game, target_value_function, state, epsilon, is_target=True)
                state = game.make_move(state, *target_action)

            # Compute the reward for the current action
            reward = 0
            if game.has_player_won(next_state, state.current_player):
                reward = 1
            if game.has_player_won(next_state, 1 - state.current_player):
                reward = -1

            # Compute the Q-value for the current state
            current_q = value_function(board_to_input(state.board, game.player_symbols).to(device))
            # Compute the Q-value for the next state if it's not terminal
            if not game.is_terminal(next_state):
                next_q_values = [
                    target_value_function(
                        board_to_input(game.make_move(next_state, row, col).board, game.player_symbols).to(device)
                    )
                    for row, col in game.get_valid_moves(next_state)
                ]
                next_q = max(next_q_values).to(device)
            else:
                next_q = torch.Tensor([[0]]).to(device)

            # Compute the target value and the loss
            target = reward + gamma * next_q
            loss = F.mse_loss(current_q, target)

            # Perform gradient descent to update the value function
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update the target network every target_update_frequency episodes
        if (episode + 1) % target_update_frequency == 0:
            target_value_function.load_state_dict(value_function.state_dict())

        # Print the progress after every 100 episodes
        if (episode + 1) % 1 == 0:
            print(f"Episode {episode + 1}/{num_episodes} completed. "
                  f"Loss: {loss.item():.4f}. "
                  f"Episode length {l}. Result: {reward}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the game environment, value function, target_value_function, and optimizer
game = GameEnvironment(winning_patterns)
value_function = ValueFunction().to(device)
target_value_function = ValueFunction().to(device)
target_value_function.load_state_dict(value_function.state_dict())
optimizer = optim.Adam(value_function.parameters(), lr=0.001)

num_episodes = 100

# Start training the value function using the defined number of episodes
train(game, value_function, target_value_function, optimizer, num_episodes)

# save model to file
torch.save(value_function.state_dict(), "model_files/value_function.pth")
