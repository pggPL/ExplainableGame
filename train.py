import torch
import torch.nn.functional as F
import torch.optim as optim
import random
from typing import Tuple, List
from model import ValueFunction, board_to_input
from env import GameEnvironment, GameState, winning_patterns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Q value is value for player 0.

# Define the ReplayBuffer class to store and sample experiences for training
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    # Add a new experience to the buffer
    def push(self,
             state: torch.Tensor,
             action_index: int,
             reward: float,
             next_state: torch.Tensor,
             done: bool) -> None:
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action_index, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    # Sample a batch of experiences from the buffer
    def sample(self, batch_size: int) -> List[Tuple]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)

# Define the epsilon-greedy policy for selecting actions during training
def epsilon_greedy_policy(
        game_environment: GameEnvironment,
        value_function: ValueFunction,
        state: GameState,
        epsilon: float,
        player: int) -> Tuple[Tuple[int, int], int]:
    valid_moves = game_environment.get_valid_moves(state)
    if random.random() < epsilon:
        chosen_action = random.choice(valid_moves)
        action_index = valid_moves.index(chosen_action)
        return chosen_action, action_index
    else:
        q_values = [
            value_function(
                board_to_input(game_environment.make_move(state, row, col).board,
                               game_environment.player_symbols).to(device)
            )
            for row, col in valid_moves
        ]
        max_index = q_values.index(max(q_values))
        min_index = q_values.index(min(q_values))
        if player == 1:
            return valid_moves[min_index], min_index
        else:
            return valid_moves[max_index], max_index

def sample_batch(
        replay_buffer: ReplayBuffer,
        game: GameEnvironment,
        value_function: ValueFunction,
        target_value_function: ValueFunction,
        batch_size: int = 1024,
        gamma: float = 0.99) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    experiences = replay_buffer.sample(batch_size)
    states, actions_indices, rewards, next_states, dones = zip(*experiences)

    states = torch.cat(states).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)
    next_states = torch.cat(next_states).to(device)
    next_q_values = target_value_function(next_states).max(1, keepdim=True)[0]
    target_q_values = rewards + (1 - dones) * gamma * next_q_values

    return states, actions_indices, target_q_values


def train(game: GameEnvironment,
          value_function: ValueFunction,
          target_value_function: ValueFunction,
          optimizer: optim.Optimizer,
          replay_buffer: ReplayBuffer,
          num_episodes: int,
          device: torch.device,
          batch_size: int = 64,
          alpha: float = 0.1,
          gamma: float = 0.99,
          epsilon_start: float = 1.0,
          epsilon_end: float = 0.1,
          epsilon_decay: float = 0.999,
          target_update_frequency: int = 10) -> None:
    for episode in range(num_episodes):
        # Initialize the state for the current episode
        state = game.initial_state()
        # Update the epsilon value for exploration
        epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** episode))
        
        # Continue until the game reaches a terminal state
        while not game.is_terminal(state):
            # Choose an action based on the epsilon-greedy policy
            action, action_index = epsilon_greedy_policy(game, value_function, state, epsilon, state.current_player)
            # Get the next state after performing the chosen action
            next_state = game.make_move(state, *action)
            
            # Compute the reward for the current action
            reward = 0
            if game.has_player_won(next_state, state.current_player):
                if state.current_player == 1:
                    reward = -1
                else:
                    reward = 1
                
            # Add the experience to the replay buffer
            state_tensor = board_to_input(state.board, game.player_symbols).unsqueeze(0).to(device)
            next_state_tensor = board_to_input(next_state.board, game.player_symbols).unsqueeze(0).to(device)
            done = game.is_terminal(next_state)
            replay_buffer.push(state_tensor, action_index, reward, next_state_tensor, done)
            
            # Update the state for the next iteration
            state = next_state
        
        loss = torch.Tensor([0])
        # Sample a batch of experiences from the replay buffer and compute the target Q-values
        if len(replay_buffer) >= batch_size:
            states, actions_indices, target_q_values = sample_batch(replay_buffer, game, value_function,
                                                                    target_value_function, batch_size)
            
            # Compute the current Q-values
            current_q_values = value_function(states)
            
            # Perform gradient descent to update the value function
            optimizer.zero_grad()
            loss = F.mse_loss(current_q_values, target_q_values)
            loss.backward()
            optimizer.step()
        
        # Update the target network every target_update_frequency episodes
        if (episode + 1) % target_update_frequency == 0:
            target_value_function.load_state_dict(value_function.state_dict())
        
        # Print the progress after every 100 episodes
        if (episode + 1) % 5 == 0:
            print(f"Episode {episode + 1}/{num_episodes} completed. Loss: {loss.item()}.")
            
            # print first position with reward in the replay buffer
            #for i in range(80):
            for i in range(len(replay_buffer)):
                if replay_buffer.buffer[i][2] != 0:
                    # Wypisz planszę
                    game_board = replay_buffer.buffer[i][0].cpu().numpy().squeeze()
                    board = game_board[0] - game_board[1]
                    for row in board:
                        print("|", end="")
                        for cell in row:
                            if cell == 1:
                                print("X|", end="")
                            elif cell == -1:
                                print("O|", end="")
                            else:
                                print(" |", end="")
                        print("\n")
                    # print action
                    print("Action: ", replay_buffer.buffer[i][1])
                    print("Reward: ", replay_buffer.buffer[i][2])

                    next_q_value = target_value_function(replay_buffer.buffer[i][3]).max(1, keepdim=True)[0]
                    
                    target_q_value = replay_buffer.buffer[i][2] + (1 - replay_buffer.buffer[i][4]) * gamma * next_q_value
                    print("Target Q-value: ",
                          target_q_value.item())
                    print("Current Q-value: ",
                          value_function(replay_buffer.buffer[i][0]).max(1, keepdim=True)[0].item())
                    print("Done: ", replay_buffer.buffer[i][4])


# Initialize the game environment, value function, target_value_function, and optimizer
game = GameEnvironment(winning_patterns)
value_function = ValueFunction().to(device)
target_value_function = ValueFunction().to(device)
target_value_function.load_state_dict(value_function.state_dict())
optimizer = optim.Adam(value_function.parameters(), lr=0.001)

# Initialize the replay buffer with a specified capacity
replay_buffer = ReplayBuffer(capacity=10000)

num_episodes = 1000

# Start training the value function using the defined number of episodes
train(game, value_function, target_value_function, optimizer, replay_buffer, num_episodes, device)

# Save the trained model to a file
torch.save(value_function.state_dict(), "model_files/value_function.pth")

