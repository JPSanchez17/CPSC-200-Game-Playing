#!/usr/bin/env python3
"""
Connect Four — Text-based Framework (no GUI)

This version keeps the general gameplay logic of the provided ConnectFour.py:
- Same board encoding (numpy 6x7, 0 empty, 1/2 players)
- Same Player API (AIPlayer/RandomPlayer/HumanPlayer from Player.py)
- Same AI timeout wrapper using multiprocessing
- Same win detection logic

Key change: Removed all non-terminal (GUI) interactions:
- No tkinter window
- No "Next Move" button
- Game runs in the terminal, printing:
    * the board
    * whose turn it is
    * the list of valid moves (columns)
- Human input happens via terminal (like the Konane framework)

Run:
  python ConnectFour.py human random
  python ConnectFour.py ai human --time 10
"""

import argparse
import multiprocessing as mp
from typing import List

import numpy as np

from Player import AIPlayer, RandomPlayer, HumanPlayer


# https://stackoverflow.com/a/37737985
def turn_worker(board, send_end, p_func):
    send_end.send(p_func(board.copy()))


def print_board(board: np.ndarray) -> None:
    """Print the Connect 4 board in a simple text format."""
    sym = {0: ".", 1: "1", 2: "2"}
    rows, cols = board.shape
    header = "    " + " ".join(f"{c:2d}" for c in range(cols))
    print(header)
    print("    " + "--" * cols)
    for r in range(rows):
        row = " ".join(f"{sym[int(board[r, c])]:2s}" for c in range(cols))
        print(f"{r:2d} | {row}")


def valid_columns(board: np.ndarray) -> List[int]:
    """Return list of columns that are not full."""
    valid = []
    for col in range(board.shape[1]):
        if 0 in board[:, col]:
            valid.append(col)
    return valid


class Game:
    def __init__(self, player1, player2, time: int, verbose: bool = True):
        self.players = [player1, player2]
        self.current_turn = 0
        self.board = np.zeros([6, 7], dtype=np.uint8)
        self.game_over = False
        self.ai_turn_limit = time
        self.verbose = verbose

        if self.verbose:
            print("\n=== Connect 4 (Text) ===")
            print_board(self.board)
            print(self._turn_banner())
            print(f"Valid moves: {valid_columns(self.board)}")

        self.play()

    def _turn_banner(self) -> str:
        p = self.players[self.current_turn]
        return f"\nPlayer {p.player_number} ({p.type}) to move"

    def play(self):
        while not self.game_over:
            self.make_move()

    def make_move(self):
        if self.game_over:
            return

        current_player = self.players[self.current_turn]

        # Choose the right AI move function (kept from original)
        if current_player.type == 'ai':
            if self.players[int(not self.current_turn)].type == 'random':
                p_func = current_player.get_expectimax_move
            else:
                p_func = current_player.get_alpha_beta_move

            try:
                recv_end, send_end = mp.Pipe(False)
                p = mp.Process(target=turn_worker, args=(self.board, send_end, p_func))
                p.start()
                if p.join(self.ai_turn_limit) is None and p.is_alive():
                    p.terminate()
                    raise Exception('Player exceeded time limit')
            except Exception as e:
                print(f'Uh oh.... something is wrong with Player {current_player.player_number}')
                print(e)
                raise Exception('Game Over')

            move = recv_end.recv()
        else:
            # Human/Random use get_move(board)
            move = current_player.get_move(self.board.copy())

        if move is None:
            raise Exception(f"Player {current_player.player_number} returned None move")

        self.update_board(int(move), current_player.player_number)

        if self.game_completed(current_player.player_number):
            self.game_over = True
            if self.verbose:
                print_board(self.board)
                print(f"\nPlayer {current_player.player_number} wins!")
            return

        # Switch turns
        self.current_turn = int(not self.current_turn)

        if self.verbose:
            print_board(self.board)
            print(self._turn_banner())
            print(f"Valid moves: {valid_columns(self.board)}")

    def update_board(self, move: int, player_num: int):
        valid = valid_columns(self.board)
        if move not in valid:
            err = f'Invalid move by player {player_num}. Column {move}. Choose from: {valid}'
            raise Exception(err)

        # Drop piece to lowest available row (kept from original logic, minus GUI updates)
        update_row = -1
        for row in range(1, self.board.shape[0]):
            update_row = -1
            if self.board[row, move] > 0 and self.board[row - 1, move] == 0:
                update_row = row - 1
            elif row == self.board.shape[0] - 1 and self.board[row, move] == 0:
                update_row = row

            if update_row >= 0:
                self.board[update_row, move] = player_num
                break

    def game_completed(self, player_num: int) -> bool:
        player_win_str = '{0}{0}{0}{0}'.format(player_num)
        board = self.board
        to_str = lambda a: ''.join(a.astype(str))

        def check_horizontal(b):
            for row in b:
                if player_win_str in to_str(row):
                    return True
            return False

        def check_verticle(b):
            return check_horizontal(b.T)

        def check_diagonal(b):
            for op in [None, np.fliplr]:
                op_board = op(b) if op else b

                root_diag = np.diagonal(op_board, offset=0).astype(int)
                if player_win_str in to_str(root_diag):
                    return True

                for i in range(1, b.shape[1] - 3):
                    for offset in [i, -i]:
                        diag = np.diagonal(op_board, offset=offset)
                        diag = to_str(diag.astype(int))
                        if player_win_str in diag:
                            return True

            return False

        return (check_horizontal(board) or
                check_verticle(board) or
                check_diagonal(board))


def main(player1: str, player2: str, time: int, verbose: bool = True):
    def make_player(name: str, num: int):
        if name == 'ai':
            return AIPlayer(num)
        elif name == 'random':
            return RandomPlayer(num)
        elif name == 'human':
            return HumanPlayer(num)
        raise ValueError('Unknown player type')

    Game(make_player(player1, 1), make_player(player2, 2), time, verbose=verbose)


if __name__ == '__main__':
    player_types = ['ai', 'random', 'human']
    parser = argparse.ArgumentParser()
    parser.add_argument('player1', choices=player_types)
    parser.add_argument('player2', choices=player_types)
    parser.add_argument('--time',
                        type=int,
                        default=60,
                        help='Time to wait for a move in seconds (int)')
    parser.add_argument('--quiet',
                        action='store_true',
                        help='Reduce printing')
    args = parser.parse_args()

    main(args.player1, args.player2, args.time, verbose=(not args.quiet))
