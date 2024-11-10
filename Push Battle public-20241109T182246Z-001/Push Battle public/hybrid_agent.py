import random
import time
import copy
from PushBattle import Game, PLAYER1, PLAYER2, EMPTY, BOARD_SIZE, NUM_PIECES, _torus

class HybridAgent:
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

        if depth == 2:
            return self.evaluate(game)
        
        possible_moves = self.get_possible_moves(game)
        if maximizing_player:
            max_eval = float('-inf')
            for move in possible_moves:
                new_game = copy.deepcopy(game)
                if len(move) == 2:
                    new_game.place_checker(move[0], move[1])
                else:
                    new_game.move_checker(move[0], move[1], move[2], move[3])

                eval = self.minimax(new_game, depth - 1, False)
                max_eval = max(max_eval, eval)
            return max_eval
        else:
            min_eval = float('inf')
            for move in possible_moves:
                new_game = copy.deepcopy(game)
                if len(move) == 2:
                    new_game.place_checker(move[0], move[1])
                else:
                    new_game.move_checker(move[0], move[1], move[2], move[3])

                eval = self.minimax(new_game, depth - 1, True)
                min_eval = min(min_eval, eval)
            return min_eval
        
    def get_heuristic(self, game, color):
        """Returns the heuristic value of a position."""
        opposite = -1
        if color == -1:
            opposite = 1
        par = 0
        tri = 0
        for i in range(8):
            for j in range(8):
                if game.board[i][j] != color:
                    continue
                if (game.board[(i+7)%8][(j+1)%8] == color):
                    par+=1
                if (game.board[i][(j+1)%8] == color):
                    par+=1
                if (game.board[(i+1)%8][(j+1)%8] == color):
                    par+=1
                if (game.board[(i+1)%8][j] == color):
                    par+=1
                # triangle
                if (game.board[(i+6)%8][j] == color and game.board[i][(j+2)%8] == color):
                    tri+=1
                if (game.board[i][(j+2)%8] == color and game.board[(i+2)%8][j] == color):
                    tri+=1
                if (game.board[(i+2)%8][j] == color and game.board[i][(j+6)%8] == color):
                    tri+=1
                if (game.board[i][(j+6)%8] == color and game.board[(i+6)%8][j] == color):
                    tri+=1
        mindist = 4
        has = []
        for i in range(8):
            for j in range(8):
                if game.board[i][j] == color:
                    has.append((i,j))
        for x in has:
            for y in has:
                if x == y:
                    continue
                dx = min(abs(x[0] - y[0]), 8 - abs(x[0] - y[0]))
                dy = min(abs(x[1] - y[1]), 8 - abs(x[1] - y[1]))
                mindist = min(mindist, max(dx, dy))
        badpar = 0
        badtri = 0
        for i in range(8):
            for j in range(8):
                if game.board[i][j] != opposite:
                    continue
                if (game.board[(i+7)%8][(j+1)%8] == opposite):
                    badpar+=1
                if (game.board[i][(j+1)%8] == opposite):
                    badpar+=1
                if (game.board[(i+1)%8][(j+1)%8] == opposite):
                    badpar+=1
                if (game.board[(i+1)%8][j] == opposite):
                    badpar+=1
                # triangle
                if (game.board[(i+6)%8][j] == opposite and game.board[i][(j+2)%8] == opposite):
                    badtri+=1
                if (game.board[i][(j+2)%8] == opposite and game.board[(i+2)%8][j] == opposite):
                    badtri+=1
                if (game.board[(i+2)%8][j] == opposite and game.board[i][(j+6)%8] == opposite):
                    badtri+=1
                if (game.board[i][(j+6)%8] == opposite and game.board[(i+6)%8][j] == opposite):
                    badtri+=1
        badmindist = 4
        has = []
        for i in range(8):
            for j in range(8):
                if game.board[i][j] == opposite:
                    has.append((i,j))
        for x in has:
            for y in has:
                if x == y:
                    continue
                dx = min(abs(x[0] - y[0]), 8 - abs(x[0] - y[0]))
                dy = min(abs(x[1] - y[1]), 8 - abs(x[1] - y[1]))
        return (3*par + tri) - (3 * badpar + badtri)
    
    def get_best_move(self, game):
        """Returns the best move using the Minimax algorithm."""
        possible_moves = self.get_possible_moves(game)
        best_move = None
        best_value = float('-inf')
        
        best_heuristic_move = None
        best_heuristic_value = float('-inf')

        for move in possible_moves:
            new_game = copy.deepcopy(game)
            if len(move) == 2:
                new_game.place_checker(move[0], move[1])
            else:
                new_game.move_checker(move[0], move[1], move[2], move[3])

            move_value = self.minimax(new_game, self.depth - 1, game.current_player == self.player)

            if game.current_player == PLAYER1:
                heuristic_value = self.get_heuristic(new_game, 1)
            else:
                heuristic_value = self.get_heuristic(new_game, -1)
            
            if heuristic_value > best_heuristic_value:
                best_heuristic_move = move
                best_heuristic_value = heuristic_value
            elif heuristic_value == best_heuristic_value:
                random.seed(time.time())
                if random.randint(1,2) % 2 == 0:
                    best_heuristic_move = move
                    best_heuristic_value = heuristic_value
            
            if move_value > best_value:
                best_value = move_value
                best_move = move
            elif move_value == best_value:
                random.seed(time.time())
                if random.randint(1,2) % 2 == 0:
                    best_value = move_value
                    best_move = move
            
        # use heuristic to find the best move
        if best_value < float('inf'):
            best_move = best_heuristic_move
        
        return best_move
