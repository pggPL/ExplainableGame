from typing import List, Tuple


class GameState:
    def __init__(self, board: List[List[str]], current_player: int):
        self.board = board
        self.current_player = current_player


class GameEnvironment:
    def __init__(self, patterns: List[List[List[int]]]):
        self.patterns = patterns
        self.player_symbols = ['X', 'O']
    
    def initial_state(self) -> GameState:
        """
        Returns the initial game state with an empty 10x10 board and the first player.
        """
        board = [['' for _ in range(10)] for _ in range(10)]
        return GameState(board, 0)
    
    def is_move_valid(self, state: GameState, row: int, col: int) -> bool:
        """
        Checks if a given move is valid by verifying that the row and column are within
        the board limits and that the cell is empty.
        """
        return 0 <= row < 10 and 0 <= col < 10 and state.board[row][col] == ''
    
    def get_valid_moves(self, state: GameState) -> List[Tuple[int, int]]:
        """
        Returns a list of valid moves as (row, col) tuples for the current state.
        """
        valid_moves = []
        for row in range(10):
            for col in range(10):
                if self.is_move_valid(state, row, col):
                    valid_moves.append((row, col))
        return valid_moves
    
    def make_move(self, state: GameState, row: int, col: int) -> GameState:
        """
        Makes a move for the current player and returns the new game state.
        """
        if self.is_move_valid(state, row, col):
            new_board = [row.copy() for row in state.board]
            new_board[row][col] = self.player_symbols[state.current_player]
            new_state = GameState(new_board, 1 - state.current_player)
            return new_state
        return state
    
    def has_player_won(self, state: GameState, player: int) -> bool:
        """
        Checks if the given player has won the game by verifying the winning patterns.
        """
        symbol = self.player_symbols[player]
        for pattern in self.patterns:
            if self.check_pattern(state, pattern, symbol):
                return True
        return False
    
    def check_pattern(self, state: GameState, pattern: List[List[int]], symbol: str) -> bool:
        """
        Verifies if the given pattern exists in the current state for the player with the given symbol.
        """
        pattern_height, pattern_width = len(pattern), len(pattern[0])
        for row in range(10 - pattern_height + 1):
            for col in range(10 - pattern_width + 1):
                matches_pattern = True
                for i, pattern_row in enumerate(pattern):
                    for j, pattern_cell in enumerate(pattern_row):
                        if pattern_cell == 1 and state.board[row + i][col + j] != symbol:
                            matches_pattern = False
                            break
                    if not matches_pattern:
                        break
                if matches_pattern:
                    return True
        return False
    
    def print_board(self, state: GameState) -> None:
        """
        Prints the current game board to the console.
        """
        print("  " + " ".join(str(i) for i in range(10)))

        for i, row in enumerate(state.board):
            print(str(i) + " " + " ".join(row))

    def is_terminal(self, state: GameState) -> bool:
        """
        Check if the current state is a terminal state.

        Returns:
            bool: True if the state is terminal, False otherwise.
        """
        # Check if either player has won
        for state.player in range(2):
            if self.has_player_won(state, state.current_player):
                return True
    
        # Check if the board is full (no more valid moves)
        if len(self.get_valid_moves(state)) == 0:
            return True
    
        # Otherwise, the state is not terminal
        return False


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
