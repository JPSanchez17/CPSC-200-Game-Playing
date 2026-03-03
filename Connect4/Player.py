import numpy as np
import math

DEFAULT_DEPTH = 5

class AIPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = 'Player {}:ai'.format(player_number)
        self.opponent_number = 2 if player_number == 1 else 1

    def get_alpha_beta_move(self, board):
        """
        Return the next move (0-based column index) using alpha-beta pruning.
        """
        _, col = self._alpha_beta(board, DEFAULT_DEPTH, -math.inf, math.inf, True)
        return col

    def get_expectimax_move(self, board):
        """
        Return the next move (0-based column index) using expectimax.
        """
        _, col = self._expectimax(board, DEFAULT_DEPTH, True)
        return col

    def evaluation_function(self, board):
        """
        Heuristic board evaluation from the perspective of self.player_number.
        Positive = good for self, negative = good for opponent.
        """
        return self._score_board(board, self.player_number) - \
               self._score_board(board, self.opponent_number)

    # Alpha-Beta (Minimax)
    def _alpha_beta(self, board, depth, alpha, beta, maximizing):
        """Returns (score, column)."""
        valid = self._valid_cols(board)

        if not valid:
            return (0, None)  # draw

        # Terminal / depth check
        if self._check_win(board, self.player_number):
            return (1_000_000 + depth, None)   # win (prefer faster wins)
        if self._check_win(board, self.opponent_number):
            return (-1_000_000 - depth, None)  # loss
        if depth == 0:
            return (self.evaluation_function(board), None)

        best_col = valid[len(valid) // 2]  # prefer center as tiebreak

        if maximizing:
            value = -math.inf
            for col in self._ordered_cols(valid, board.shape[1]):
                new_board = self._drop(board, col, self.player_number)
                score, _ = self._alpha_beta(new_board, depth - 1, alpha, beta, False)
                if score > value:
                    value, best_col = score, col
                alpha = max(alpha, value)
                if alpha >= beta:
                    break  # beta cut-off
            return (value, best_col)
        else:
            value = math.inf
            for col in self._ordered_cols(valid, board.shape[1]):
                new_board = self._drop(board, col, self.opponent_number)
                score, _ = self._alpha_beta(new_board, depth - 1, alpha, beta, True)
                if score < value:
                    value, best_col = score, col
                beta = min(beta, value)
                if alpha >= beta:
                    break  # alpha cut-off
            return (value, best_col)

    # Expectimax
    def _expectimax(self, board, depth, maximizing):
        """Returns (score, column)."""
        valid = self._valid_cols(board)

        if not valid:
            return (0, None)

        if self._check_win(board, self.player_number):
            return (1_000_000 + depth, None)
        if self._check_win(board, self.opponent_number):
            return (-1_000_000 - depth, None)
        if depth == 0:
            return (self.evaluation_function(board), None)

        best_col = valid[len(valid) // 2]

        if maximizing:
            value = -math.inf
            for col in self._ordered_cols(valid, board.shape[1]):
                new_board = self._drop(board, col, self.player_number)
                score, _ = self._expectimax(new_board, depth - 1, False)
                if score > value:
                    value, best_col = score, col
            return (value, best_col)
        else:
            # Chance node: opponent is random, uniform over valid moves
            total = 0.0
            prob = 1.0 / len(valid)
            for col in valid:
                new_board = self._drop(board, col, self.opponent_number)
                score, _ = self._expectimax(new_board, depth - 1, True)
                total += prob * score
            return (total, None)

    # Board helpers
    def _valid_cols(self, board):
        return [c for c in range(board.shape[1]) if 0 in board[:, c]]

    def _ordered_cols(self, valid, width):
        """Return valid columns ordered by distance from center (prefer center)."""
        center = width // 2
        return sorted(valid, key=lambda c: abs(c - center))

    def _drop(self, board, col, player_num):
        """Return a new board with player_num's piece dropped in col."""
        new_board = board.copy()
        for row in range(board.shape[0] - 1, -1, -1):
            if new_board[row, col] == 0:
                new_board[row, col] = player_num
                break
        return new_board

    def _check_win(self, board, player_num):
        win_str = '{0}{0}{0}{0}'.format(player_num)
        to_str = lambda a: ''.join(a.astype(str))

        # Horizontal
        for row in board:
            if win_str in to_str(row):
                return True
        # Vertical
        for col in board.T:
            if win_str in to_str(col):
                return True
        # Diagonals
        for op in [None, np.fliplr]:
            b = op(board) if op else board
            for offset in range(-(board.shape[0] - 4), board.shape[1] - 3):
                diag = np.diagonal(b, offset=offset)
                if len(diag) >= 4 and win_str in to_str(diag.astype(int)):
                    return True
        return False

    # Evaluation / scoring helpers
    def _score_board(self, board, player_num):
        """Sum heuristic scores for all windows of 4 for the given player."""
        score = 0

        # Center column preference
        center_col = board[:, board.shape[1] // 2]
        score += int(np.sum(center_col == player_num)) * 3

        # Horizontal
        for row in range(board.shape[0]):
            for col in range(board.shape[1] - 3):
                window = board[row, col:col + 4]
                score += self._score_window(window, player_num)

        # Vertical
        for col in range(board.shape[1]):
            for row in range(board.shape[0] - 3):
                window = board[row:row + 4, col]
                score += self._score_window(window, player_num)

        # Diagonal (positive slope)
        for row in range(board.shape[0] - 3):
            for col in range(board.shape[1] - 3):
                window = [board[row + i, col + i] for i in range(4)]
                score += self._score_window(np.array(window), player_num)

        # Diagonal (negative slope)
        for row in range(3, board.shape[0]):
            for col in range(board.shape[1] - 3):
                window = [board[row - i, col + i] for i in range(4)]
                score += self._score_window(np.array(window), player_num)

        return score

    def _score_window(self, window, player_num):
        """Score a window of 4 cells."""
        opp = 2 if player_num == 1 else 1
        p_count = int(np.sum(window == player_num))
        o_count = int(np.sum(window == opp))
        empty = int(np.sum(window == 0))

        if o_count > 0:
            return 0  # mixed window, no value

        if p_count == 4:
            return 100
        elif p_count == 3 and empty == 1:
            return 5
        elif p_count == 2 and empty == 2:
            return 2
        return 0


class RandomPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'random'
        self.player_string = 'Player {}:random'.format(player_number)

    def get_move(self, board):
        valid_cols = []
        for col in range(board.shape[1]):
            if 0 in board[:, col]:
                valid_cols.append(col)
        return int(np.random.choice(valid_cols))


class HumanPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'human'
        self.player_string = 'Player {}:human'.format(player_number)

    def get_move(self, board):
        valid_cols = []
        for i, col in enumerate(board.T):
            if 0 in col:
                valid_cols.append(i)

        print(f"[q] quit")

        raw = input('Enter your move: ').strip()
        if raw.lower() in ('q', 'quit', 'exit'):
            raise SystemExit('Player chose to quit.')

        move = int(raw)

        while move not in valid_cols:
            print('Column full/invalid, choose from: {}'.format(valid_cols))
            raw = input('Enter your move (or q to quit): ').strip()
            if raw.lower() in ('q', 'quit', 'exit'):
                raise SystemExit('Player chose to quit.')
            move = int(raw)

        return move