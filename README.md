# CPSC-200-Game-Playing
CPSC 200 Assignment - Connect 4

# Connect Four AI Implementation Report

## Alpha-Beta Pruning

This algorithm looks several moves ahead and assumes the opponent will always make the best possible move. The AI tries to maximize its own score while the opponent tries to minimize it. Branches of the search that cannot possibly change the outcome are skipped, which makes the search much faster.

**Variations from the standard algorithm:**
- The AI is rewarded more for winning quickly and penalized less for losing slowly. This means it will try to end the game as soon as possible when it is winning, rather than being indifferent about timing.
- The AI always considers center columns before edge columns, since center positions tend to be stronger in Connect Four. Checking better moves first helps the pruning skip more of the search tree.

---

## Expectimax

Expectimax works similarly but does not assume the opponent plays perfectly. Instead, it treats the opponent as if they pick a random move each turn and calculates the average outcome across all those possibilities. This makes it a better fit for opponents who are unpredictable or do not always make the best move.

**Variations from the standard algorithm:**
- Because outcomes are averaged rather than compared as best or worst case, no branches can be safely skipped the way alpha-beta does. Every possible opponent move has to be considered.
- Move ordering is only applied on the AI's turn. On the opponent's turn, order does not matter since all moves are weighted equally in the average.

---

## Static Evaluation Function

When the AI cannot search all the way to the end of the game, it uses the evaluation function to judge how good the current board looks. It scans every group of 4 consecutive cells on the board, checking rows, columns, and diagonals.

Each group is scored based on how many of the AI's pieces are in it:

| Window Contents | Score |
|---|---|
| 4 AI pieces | 100 |
| 3 AI pieces + 1 empty | 5 |
| 2 AI pieces + 2 empty | 2 |
| Any opponent piece present | 0 |

A group that contains any opponent piece scores 0 since it is already blocked and cannot lead to a win. The AI also gets a small bonus for pieces placed in the center column, since the center is the most flexible position on the board.

The final score is the AI's total minus the opponent's total, so the function rewards strong positions for the AI while penalizing strong positions for the opponent.

---

## AI Support

AI support was used to provide basic sample code to understand how to implement minimax and alpha-beta pruning into the program.
