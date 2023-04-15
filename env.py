class GameEnvironment:
    def __init__(self, patterns):
        self.board = [[' ' for _ in range(10)] for _ in range(10)]
        self.patterns = patterns
        self.player_symbols = ['X', 'O']

    def is_move_valid(self, row, col):
        return 0 <= row < 10 and 0 <= col < 10 and self.board[row][col] == ' '

    def get_valid_moves(self):
        valid_moves = []
        for row in range(10):
            for col in range(10):
                if self.is_move_valid(row, col):
                    valid_moves.append((row, col))
        return valid_moves

    def make_move(self, row, col, player):
        if self.is_move_valid(row, col):
            self.board[row][col] = self.player_symbols[player]
            return True
        return False

    def get_board_representation(self):
        return self.board

    def has_player_won(self, player):
        symbol = self.player_symbols[player]
        for pattern in self.patterns:
            if self.check_pattern(pattern, symbol):
                return True
        return False

    def check_pattern(self, pattern, symbol):
        pattern_height, pattern_width = len(pattern), len(pattern[0])
        for row in range(10 - pattern_height + 1):
            for col in range(10 - pattern_width + 1):
                matches_pattern = True
                for i, pattern_row in enumerate(pattern):
                    for j, pattern_cell in enumerate(pattern_row):
                        if pattern_cell == 1 and self.board[row+i][col+j] != symbol:
                            matches_pattern = False
                            break
                    if not matches_pattern:
                        break
                if matches_pattern:
                    return True
        return False

    def print_board(self):
        print("  " + " ".join(str(i) for i in range(10)))
        for i, row in enumerate(self.board):
            print(str(i) + " " + " ".join(row))

# Przykład użycia:
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

game = GameEnvironment(winning_patterns)
game.print_board()