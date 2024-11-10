import random
import numpy as np
import pickle
import os
from datetime import datetime
from PushBattle import Game, PLAYER1, PLAYER2, EMPTY, BOARD_SIZE, NUM_PIECES, _torus
from random_agent import RandomAgent

class QLearningAgent:
    def __init__(self, player=PLAYER1, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.player = player
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = {}
        self.training_stats = {
            'episodes_trained': 0,
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'creation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def save(self, filepath=None):
        """
        Save the trained agent to a file.
        If no filepath is provided, creates a timestamped file in the current directory.
        """
        if filepath is None:
            # Create a directory for saved models if it doesn't exist
            os.makedirs('saved_models', exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f'saved_models/qagent_{timestamp}.pkl'
        
        save_data = {
            'q_table': self.q_table,
            'player': self.player,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'training_stats': self.training_stats
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        return filepath
    
    @classmethod
    def load(cls, filepath):
        """
        Load a trained agent from a file.
        Returns a new QLearningAgent instance with the loaded data.
        """
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        # Create a new instance with the saved parameters
        agent = cls(
            player=save_data['player'],
            learning_rate=save_data['learning_rate'],
            discount_factor=save_data['discount_factor'],
            epsilon=save_data['epsilon']
        )
        
        # Load the saved Q-table and training stats
        agent.q_table = save_data['q_table']
        agent.training_stats = save_data['training_stats']
        
        return agent
    
    def get_state_key(self, game):
        """Convert the game state into a hashable key for the Q-table."""
        board_state = tuple(tuple(row) for row in game.board)
        return (board_state, game.p1_pieces, game.p2_pieces, game.current_player)
    
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

    def get_q_value(self, state_key, action):
        """Get Q-value for a state-action pair. Initialize if not exists."""
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        if action not in self.q_table[state_key]:
            self.q_table[state_key][action] = 0.0
        return self.q_table[state_key][action]

    def get_reward(self, game):
        """Calculate reward based on game state."""
        winner = game.check_winner()
        if winner == self.player:
            self.training_stats['wins'] += 1
            return 1.0
        elif winner == EMPTY:
            # Small positive reward for controlling center positions
            reward = 0.0
            center = BOARD_SIZE // 2
            for r in range(center-1, center+2):
                for c in range(center-1, center+2):
                    if game.board[r][c] == self.player:
                        reward += 0.1
                    elif game.board[r][c] != EMPTY:
                        reward -= 0.1
            return reward
        else:
            self.training_stats['losses'] += 1
            return -1.0

    def choose_action(self, game, training=True):
        """Choose action using epsilon-greedy policy."""
        possible_moves = self.get_possible_moves(game)
        state_key = self.get_state_key(game)
        
        if training and random.random() < self.epsilon:
            return random.choice(possible_moves)
        
        best_value = float('-inf')
        best_moves = []
        
        for move in possible_moves:
            q_value = self.get_q_value(state_key, move)
            if q_value > best_value:
                best_value = q_value
                best_moves = [move]
            elif q_value == best_value:
                best_moves.append(move)
        
        return random.choice(best_moves)

    def learn(self, state, action, reward, next_state, next_possible_moves):
        """Update Q-values using Q-learning algorithm."""
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        current_q = self.get_q_value(state_key, action)
        
        next_max_q = float('-inf')
        for next_move in next_possible_moves:
            next_q = self.get_q_value(next_state_key, next_move)
            next_max_q = max(next_max_q, next_q)
        
        if not next_possible_moves:
            next_max_q = 0
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * next_max_q - current_q
        )
        
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        self.q_table[state_key][action] = new_q

    def get_best_move(self, game):
        """Interface method to match other agents."""
        return self.choose_action(game, training=False)

    def train(self, num_episodes=1000, opponent=None, save_interval=None):
        """
        Train the agent through self-play or against a given opponent.
        
        Args:
            num_episodes: Number of games to play
            opponent: Opponent agent (if None, uses AlphaBetaAgent)
            save_interval: If provided, saves the model every save_interval episodes
        """
        if opponent is None:
            opponent = RandomAgent(PLAYER2 if self.player == PLAYER1 else PLAYER1)
        
        start_time = datetime.now()
        
        for episode in range(num_episodes):
            game = Game()
            game_history = []
            
            while game.check_winner() == EMPTY:
                current_state = game
                
                if game.current_player == self.player:
                    action = self.choose_action(game, training=True)
                    game_history.append((current_state, action))
                    
                    if len(action) == 2:
                        game.place_checker(action[0], action[1])
                    else:
                        game.move_checker(action[0], action[1], action[2], action[3])
                else:
                    opponent_move = opponent.get_best_move(game)
                    if len(opponent_move) == 2:
                        game.place_checker(opponent_move[0], opponent_move[1])
                    else:
                        game.move_checker(opponent_move[0], opponent_move[1], 
                                        opponent_move[2], opponent_move[3])
            
            if game.check_winner() == EMPTY:
                self.training_stats['draws'] += 1
                
            final_reward = self.get_reward(game)
            
            for i, (state, action) in enumerate(game_history):
                next_state = game_history[i + 1][0] if i + 1 < len(game_history) else game
                next_moves = self.get_possible_moves(next_state)
                reward = final_reward * (self.discount_factor ** (len(game_history) - i - 1))
                self.learn(state, action, reward, next_state, next_moves)

            self.epsilon = max(0.01, self.epsilon * 0.995)
            self.training_stats['episodes_trained'] += 1
            
            # Save periodically if requested
            if save_interval and (episode + 1) % save_interval == 0:
                self.save(f'saved_models/qagent_checkpoint_{episode+1}.pkl')
        
        # Update training statistics
        self.training_stats['training_duration'] = str(datetime.now() - start_time)
        
        # Save final model
        final_path = self.save()
        return final_path