# Pop Tac Toe AI Bots

This repository contains the code for our AI bots developed to compete in the **Push Battle** challenge at **TAMU Datathon 2024**.

## Game Rules

**Pop Tac Toe** is a two-player game played on an 8x8 grid. The main rules are as follows:

1. **Piece Placement**: Each player has 8 pieces. Players take turns placing their pieces on the board.
2. **Piece Pushing**: When a piece is placed, all adjacent pieces are pushed away (unless blocked).
3. **Edge Wrapping**: Pieces pushed off the board reappear on the opposite side.
4. **Moving Pieces**: If all 8 pieces are on the board, a player must pick up one of their pieces and move it to any other spot.
5. **Winning Condition**: The objective is to get 3 of your pieces in a row (horizontally, vertically, or diagonally).

## Gameplay Restrictions

- **Time Limit**: Players have 5 seconds per move. Exceeding this time grants an additional 5 seconds for a fallback move. After 5 timeouts, the player forfeits.
- **Invalid Moves**: An invalid move (e.g., placing off-board, on an existing piece, or moving the opponent's piece) results in an immediate forfeit.
- **Resource Limits**: Bots are restricted to 1 CPU core, 1 GB RAM, and 1 GB VRAM. No network access is allowed.

## Algorithms Used

Our bots use a combination of heuristic algorithms, minimax search with alpha-beta pruning, and attempts at reinforcement learning:

- **Heuristic Algorithms**: We created a set of rules and evaluation functions for making generally good moves, even if they aren’t always optimal.
- **Minimax with Alpha-Beta Pruning**: We implemented a minimax algorithm with a depth of 2, allowing our bot to anticipate "mate in 1" situations. Alpha-beta pruning optimizes the search by eliminating non-optimal branches.
- **Reinforcement Learning (Q-Learning)**: Initially, we explored Q-Learning to train the bot. However, the training time required to play at a competitive level was impractical for the datathon.

## The Winning Bot: Hybrid Agent 2

The bot that won first prize in the competition is implemented in `hybrid_agent2.py`. This agent combines heuristic approaches with minimax search to depth 2, enabling strategic moves and avoiding immediate losses.

---

Enjoy exploring our code and feel free to experiment with and improve upon our AI! We would love to hear your feedback.
