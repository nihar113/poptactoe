import random
from PushBattle import Game, PLAYER1, PLAYER2, EMPTY, BOARD_SIZE, NUM_PIECES, _torus

class MinimaxAgent:
    def __init__(self, player=PLAYER1, depth=3):
        self.player = player  # the agent's player
        self.depth = depth    # depth for the Minimax search
        
    def get_possible_moves(self, game):
        """Returns list of all possible moves in current state."""
        moves = []
        current_pieces = game.p1_pieces if game.current_player == PLAYER1 else game.p2_pieces
        
        if current_pieces < NUM_PIECES:
            # placement moves
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    if game.board[r][c] == EMPTY:
                        moves.append((r, c))
        else:
            # movement moves
            for r0 in range(BOARD_SIZE):
                for c0 in range(BOARD_SIZE):
                    if game.board[r0][c0] == game.current_player:
                        for r1 in range(BOARD_SIZE):
                            for c1 in range(BOARD_SIZE):
                                if game.board[r1][c1] == EMPTY:
                                    moves.append((r0, c0, r1, c1))
        return moves

    def evaluate(self, game):
        """Simple evaluation function. Higher scores are better for the agent."""
        # Example: Consider number of pieces the agent has and its opponent has
        if game.current_player == self.player:
            return game.p1_pieces - game.p2_pieces
        else:
            return game.p2_pieces - game.p1_pieces

    def minimax(self, game, depth, maximizing_player):
        """Minimax algorithm with alpha-beta pruning."""
        winner = game.check_winner()
        if winner != EMPTY:
            if winner == game.current_player:
                if maximizing_player:
                    return float('inf')
                else:
                    return float('-inf')
            else:
                if maximizing_player:
                    return float('-inf')
                else:
                    return float('inf')

        if depth == 0:
            return self.evaluate(game)
        
        possible_moves = self.get_possible_moves(game)
        if maximizing_player:
            max_eval = float('-inf')
            for move in possible_moves:
                new_game = game  # make a copy of the current game state
                if len(move) == 2:
                    new_game.place_checker(move[0], move[1])
                else:
                    new_game.move_checker(move[0], move[1], move[2], move[3])

                eval = self.minimax(new_game, depth - 1, False)  # minimize for the opponent
                max_eval = max(max_eval, eval)
            return max_eval
        else:
            min_eval = float('inf')
            for move in possible_moves:
                new_game = game  # make a copy of the current game state
                if len(move) == 2:
                    new_game.place_checker(move[0], move[1])
                else:
                    new_game.move_checker(move[0], move[1], move[2], move[3])

                eval = self.minimax(new_game, depth - 1, True)  # maximize for the agent
                min_eval = min(min_eval, eval)
            return min_eval
    
    def get_best_move(self, game):
        """Returns the best move using the Minimax algorithm."""
        possible_moves = self.get_possible_moves(game)
        best_move = None
        best_value = float('-inf') if game.current_player == self.player else float('inf')
        
        for move in possible_moves:
            new_game = game  # make a copy of the current game state
            if len(move) == 2:
                new_game.place_checker(move[0], move[1])
            else:
                new_game.move_checker(move[0], move[1], move[2], move[3])

            move_value = self.minimax(new_game, self.depth - 1, game.current_player == self.player)
            
            if (game.current_player == self.player and move_value > best_value) or \
               (game.current_player != self.player and move_value < best_value):
                best_value = move_value
                best_move = move
        
        return best_move
