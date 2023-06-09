import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class ValueFunction(nn.Module):
    def __init__(self):
        super(ValueFunction, self).__init__()
        # Define the layers of the neural network
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 10 * 10, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass the input tensor through the layers of the neural network
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64 * 10 * 10)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def board_to_input(board: List[List[str]], player_symbols: Tuple[str, str]) -> torch.Tensor:
    """
    Convert the game board to an input tensor for the neural network.

    Args:
        board (List[List[str]]): The game board represented as a list of lists.
        player_symbols (Tuple[str, str]): The symbols used by the players, usually ('X', 'O').

    Returns:
        torch.Tensor: The input tensor for the neural network.
    """
    input_tensor = torch.zeros(3, 10, 10, dtype=torch.float32)
    for i, row in enumerate(board):
        for j, cell in enumerate(row):
            if cell == player_symbols[0]:
                input_tensor[0, i, j] = 1
            elif cell == player_symbols[1]:
                input_tensor[1, i, j] = 1
            else:
                input_tensor[2, i, j] = 1
    return input_tensor
