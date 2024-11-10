import random
from PushBattle import Game, PLAYER1, PLAYER2, EMPTY, BOARD_SIZE, NUM_PIECES, _torus

class AlphaBetaAgent:
    def __init__(self, player=PLAYER1, depth=3):
        self.player = player
        self.depth = depth

    def get_possible_moves(self, game):
        """Returns list of all possible moves in the current state."""
        moves = []
        current_pieces = game.p1_pieces if game.current_player == PLAYER1 else game.p2_pieces

        if current_pieces < NUM_PIECES:
            # Placement moves
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    if game.board[r][c] == EMPTY:
                        moves.append((r, c))
        else:
            # Movement moves
            for r0 in range(BOARD_SIZE):
                for c0 in range(BOARD_SIZE):
                    if game.board[r0][c0] == game.current_player:
                        for r1 in range(BOARD_SIZE):
                            for c1 in range(BOARD_SIZE):
                                if game.board[r1][c1] == EMPTY:
                                    moves.append((r0, c0, r1, c1))
        return moves

    def evaluate(self, game):
        """Enhanced evaluation function to assess board strength."""
        score = 0
        # Points for number of pieces
        if game.current_player == self.player:
            score += game.p1_pieces - game.p2_pieces
        else:
            score += game.p2_pieces - game.p1_pieces

        # Points for control of the center area
        center = BOARD_SIZE // 2
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if game.board[r][c] == self.player:
                    score += 2 if abs(r - center) <= 1 and abs(c - center) <= 1 else 1
                elif game.board[r][c] != EMPTY:
                    score -= 2 if abs(r - center) <= 1 and abs(c - center) <= 1 else 1

        return score

    def alpha_beta(self, game, depth, alpha, beta, maximizing_player):
        """Minimax algorithm with alpha-beta pruning."""
        winner = game.check_winner()
        if winner != EMPTY:
            return float('inf') if winner == self.player else float('-inf')
        
        if depth == 0:
            return self.evaluate(game)
        
        possible_moves = self.get_possible_moves(game)
        if maximizing_player:
            max_eval = float('-inf')
            for move in possible_moves:
                new_game = game  # use clone to create a copy of the game state
                if len(move) == 2:
                    new_game.place_checker(move[0], move[1])
                else:
                    new_game.move_checker(move[0], move[1], move[2], move[3])

                eval = self.alpha_beta(new_game, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # beta cut-off
            return max_eval
        else:
            min_eval = float('inf')
            for move in possible_moves:
                new_game = game
                if len(move) == 2:
                    new_game.place_checker(move[0], move[1])
                else:
                    new_game.move_checker(move[0], move[1], move[2], move[3])

                eval = self.alpha_beta(new_game, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # alpha cut-off
            return min_eval

    def get_best_move(self, game):
        """Returns the best move using the Alpha-Beta pruning algorithm."""
        possible_moves = self.get_possible_moves(game)
        best_move = None
        best_value = float('-inf') if game.current_player == self.player else float('inf')

        for move in possible_moves:
            new_game = game
            if len(move) == 2:
                new_game.place_checker(move[0], move[1])
            else:
                new_game.move_checker(move[0], move[1], move[2], move[3])

            move_value = self.alpha_beta(new_game, self.depth - 1, float('-inf'), float('inf'), game.current_player == self.player)

            if (game.current_player == self.player and move_value > best_value) or \
               (game.current_player != self.player and move_value < best_value):
                best_value = move_value
                best_move = move

        return best_move
