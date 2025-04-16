# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
import random
import math

COLOR_MAP = {
    0: "#cdc1b4", 2: "#eee4da", 4: "#ede0c8", 8: "#f2b179",
    16: "#f59563", 32: "#f67c5f", 64: "#f65e3b", 128: "#edcf72",
    256: "#edcc61", 512: "#edc850", 1024: "#edc53f", 2048: "#edc22e",
    4096: "#3c3a32", 8192: "#3c3a32", 16384: "#3c3a32", 32768: "#3c3a32"
}
TEXT_COLOR = {
    2: "#776e65", 4: "#776e65", 8: "#f9f6f2", 16: "#f9f6f2",
    32: "#f9f6f2", 64: "#f9f6f2", 128: "#f9f6f2", 256: "#f9f6f2",
    512: "#f9f6f2", 1024: "#f9f6f2", 2048: "#f9f6f2", 4096: "#f9f6f2"
}

class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()

        self.size = 4  # 4x4 2048 board
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0

        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]

        self.last_move_valid = True  # Record if the last move was valid

        self.reset()

    def reset(self):
        """Reset the environment"""
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board

    def add_random_tile(self):
        """Add a random tile (2 or 4) to an empty cell"""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x, y] = 2 if random.random() < 0.9 else 4

    def compress(self, row):
        """Compress the row: move non-zero values to the left"""
        new_row = row[row != 0]  # Remove zeros
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')  # Pad with zeros on the right
        return new_row

    def merge(self, row):
        """Merge adjacent equal numbers in the row"""
        for i in range(len(row) - 1):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                row[i + 1] = 0
                self.score += row[i]
        return row

    def move_left(self):
        """Move the board left"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            new_row = self.compress(self.board[i])
            new_row = self.merge(new_row)
            new_row = self.compress(new_row)
            self.board[i] = new_row
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_right(self):
        """Move the board right"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            # Reverse the row, compress, merge, compress, then reverse back
            reversed_row = self.board[i][::-1]
            reversed_row = self.compress(reversed_row)
            reversed_row = self.merge(reversed_row)
            reversed_row = self.compress(reversed_row)
            self.board[i] = reversed_row[::-1]
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_up(self):
        """Move the board up"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            col = self.compress(self.board[:, j])
            col = self.merge(col)
            col = self.compress(col)
            self.board[:, j] = col
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def move_down(self):
        """Move the board down"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            # Reverse the column, compress, merge, compress, then reverse back
            reversed_col = self.board[:, j][::-1]
            reversed_col = self.compress(reversed_col)
            reversed_col = self.merge(reversed_col)
            reversed_col = self.compress(reversed_col)
            self.board[:, j] = reversed_col[::-1]
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def is_game_over(self):
        """Check if there are no legal moves left"""
        # If there is any empty cell, the game is not over
        if np.any(self.board == 0):
            return False

        # Check horizontally
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j+1]:
                    return False

        # Check vertically
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.board[i, j] == self.board[i+1, j]:
                    return False

        return True

    def step(self, action):
        """Execute one action"""
        assert self.action_space.contains(action), "Invalid action"

        if action == 0:
            moved = self.move_up()
        elif action == 1:
            moved = self.move_down()
        elif action == 2:
            moved = self.move_left()
        elif action == 3:
            moved = self.move_right()
        else:
            moved = False

        self.last_move_valid = moved  # Record if the move was valid

        if moved:
            self.add_random_tile()

        done = self.is_game_over()

        return self.board, self.score, done, {}

    def render(self, mode="human", action=None):
        """
        Render the current board using Matplotlib.
        This function does not check if the action is valid and only displays the current board state.
        """
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)

        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i, j]
                color = COLOR_MAP.get(value, "#3c3a32")  # Default dark color
                text_color = TEXT_COLOR.get(value, "white")
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor="black")
                ax.add_patch(rect)

                if value != 0:
                    ax.text(j, i, str(value), ha='center', va='center',
                            fontsize=16, fontweight='bold', color=text_color)
        title = f"score: {self.score}"
        if action is not None:
            title += f" | action: {self.actions[action]}"
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.show()

    def simulate_row_move(self, row):
        """Simulate a left move for a single row"""
        # Compress: move non-zero numbers to the left
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        # Merge: merge adjacent equal numbers (do not update score)
        for i in range(len(new_row) - 1):
            if new_row[i] == new_row[i + 1] and new_row[i] != 0:
                new_row[i] *= 2
                new_row[i + 1] = 0
        # Compress again
        new_row = new_row[new_row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def is_move_legal(self, action):
        """Check if the specified move is legal (i.e., changes the board)"""
        # Create a copy of the current board state
        temp_board = self.board.copy()

        if action == 0:  # Move up
            for j in range(self.size):
                col = temp_board[:, j]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col
        elif action == 1:  # Move down
            for j in range(self.size):
                # Reverse the column, simulate, then reverse back
                col = temp_board[:, j][::-1]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col[::-1]
        elif action == 2:  # Move left
            for i in range(self.size):
                row = temp_board[i]
                temp_board[i] = self.simulate_row_move(row)
        elif action == 3:  # Move right
            for i in range(self.size):
                row = temp_board[i][::-1]
                new_row = self.simulate_row_move(row)
                temp_board[i] = new_row[::-1]
        else:
            raise ValueError("Invalid action")

        # If the simulated board is different from the current board, the move is legal
        return not np.array_equal(self.board, temp_board)

from collections import defaultdict
import numpy as np
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
import random
import math
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from time import sleep

class NTupleApproximator:
    def __init__(self, board_size, patterns):
        """
        Initializes the N-Tuple approximator.
        """
        self.board_size = board_size
        self.patterns = patterns
        self.weights = [defaultdict(float) for _ in patterns]
        self.symmetry_patterns = []
        for idx, pattern in enumerate(self.patterns):
            syms = self.generate_symmetries(pattern)
            for syms_ in syms:
                self.symmetry_patterns.append((idx, syms_))

    def generate_symmetries(self, pattern):
        board_size = self.board_size
        symmetries = []

        def transform_pattern(pattern, transform_func):
            return [transform_func(i, j) for (i, j) in pattern]

        def rot90(i, j):
            return (j, board_size - 1 - i)

        def rot180(i, j):
            return (board_size - 1 - i, board_size - 1 - j)

        def rot270(i, j):
            return (board_size - 1 - j, i)

        def reflect_horiz(i, j):
            return (board_size - 1 - i, j)

        symmetries.append(pattern)
        symmetries.append(transform_pattern(pattern, rot90))
        symmetries.append(transform_pattern(pattern, rot180))
        symmetries.append(transform_pattern(pattern, rot270))
        symmetries.append(transform_pattern(pattern, reflect_horiz))
        symmetries.append(transform_pattern(transform_pattern(pattern, rot90), reflect_horiz))
        symmetries.append(transform_pattern(transform_pattern(pattern, rot180), reflect_horiz))
        symmetries.append(transform_pattern(transform_pattern(pattern, rot270), reflect_horiz))

        return symmetries

    def tile_to_index(self, tile):
        if tile == 0:
            return 0
        else:
            return int(math.log(tile, 2))

    def get_feature(self, board, coords):
        feature = []
        for coord in coords:
            i, j = coord
            feature.append(self.tile_to_index(board[i, j]))
        return tuple(feature)


    def value(self, board):
        value = 0
        for idx, pattern in self.symmetry_patterns:
            feature = self.get_feature(board, pattern)
            pattern_value = self.weights[idx][feature]
            value += pattern_value
        return value

    def update(self, board, delta, alpha):
        for idx, pattern in self.symmetry_patterns:
            feature = self.get_feature(board, pattern)
            self.weights[idx][feature] += alpha * delta / len(self.patterns)

import pickle
# Define n-tuple patterns
patterns = [
    [(0, 0), (0, 1), (0, 2), (0, 3)],
    [(1, 0), (1, 1), (1, 2), (1, 3)],
    [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
    [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)],
    [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)],
    [(1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1)],
    [(2, 0), (2, 1), (2, 2), (2, 3), (3, 0), (3, 1)],
]

env = Game2048Env()


class TD_MCTS_Node:
    def __init__(self, state, score, parent=None, action=None):
        """
        state: Current board state (numpy array).
        score: Cumulative score at this node.
        parent: Parent node (None for root).
        action: Action taken from parent to reach this node.
        """
        self.state = state
        self.score = score
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        self.untried_actions = [a for a in range(4) if env.is_move_legal(a)]

    def fully_expanded(self):
        return len(self.untried_actions) == 0

class TD_MCTS:
    def __init__(self, env, approximator, iterations=500, exploration_constant=1.41, rollout_depth=10, gamma=0.99):
        """
        Initialize TD-MCTS.
        
        Args:
            env: Environment with methods is_move_legal(action), step(action), is_game_over(),
                 and attributes board (numpy array) and score (float/int).
            approximator: Value approximator with method value(state) -> float.
            iterations: Number of MCTS iterations.
            exploration_constant: UCT exploration parameter.
            rollout_depth: Maximum depth for random rollouts.
            gamma: Discount factor for backpropagation.
        """
        self.env = env
        self.approximator = approximator
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.gamma = gamma

    def create_env_from_state(self, state, score):
        try:
            new_env = copy.deepcopy(self.env)
            new_env.board = state.copy()
            new_env.score = score
            return new_env
        except AttributeError as e:
            raise ValueError("Environment must have 'board' and 'score' attributes") from e

    def select_child(self, node):
        best_action = None
        best_score = -float('inf')

        for action, child in node.children.items():
            if child.visits == 0:
                return action
            uct_score = (child.total_reward / child.visits) + self.c * np.sqrt(np.log(node.visits + 1) / child.visits)
            if uct_score > best_score:
                best_score = uct_score
                best_action = action

        return best_action

    def rollout(self, sim_env, depth):
        current_depth = 0
        while current_depth < depth and not sim_env.is_game_over():
            legal_moves = [a for a in range(4) if sim_env.is_move_legal(a)]
            if not legal_moves:
                break
            action = random.choice(legal_moves)
            sim_env.step(action)
            current_depth += 1
        return self.approximator.value(sim_env.board)

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def run_simulation(self, root):
        node = root
        sim_env = self.create_env_from_state(node.state, node.score)

        # Selection
        while node.fully_expanded() and not sim_env.is_game_over():
            action = self.select_child(node)
            node = node.children[action]
            sim_env.step(action)
            if sim_env.is_game_over():
                break

        # Expansion
        if node.untried_actions and not sim_env.is_game_over():
            action = random.choice(node.untried_actions)
            node.untried_actions.remove(action)
            new_state, new_score, done, _ = sim_env.step(action)
            node.children[action] = TD_MCTS_Node(new_state, new_score, parent=node, action=action)
            node = node.children[action]

        # Rollout
        rollout_reward = self.rollout(sim_env, self.rollout_depth)
        # Backpropagate
        self.backpropagate(node, rollout_reward)

    def best_action_distribution(self, root):
        total_visits = sum(child.visits for child in root.children.values())
        distribution = np.zeros(4)
        best_visits = -1
        best_action = None

        for action, child in root.children.items():
            visits = child.visits
            distribution[action] = visits / total_visits if total_visits > 0 else 0
            if visits > best_visits:
                best_visits = visits
                best_action = action

        return best_action, distribution

    def search(self, state, score):
        """
        Run MCTS to find the best action from the given state.
        
        Args:
            state: Initial board state (numpy array).
            score: Initial score.
        
        Returns:
            best_action: Best action to take.
            distribution: Normalized visit count distribution over actions.
        """
        root = TD_MCTS_Node(state, score)
        for _ in range(self.iterations):
            self.run_simulation(root)
        best_action, distribution = self.best_action_distribution(root)
        return best_action, distribution


with open('approximator1.pkl', 'rb') as f:
    approximator = pickle.load(f)

def get_action(state, score):
    global env
    env.board = state.copy()
    env.score = score

    mcts = TD_MCTS(
        env=env,
        approximator=approximator,  
        iterations=100,  
        exploration_constant=1.41, 
        rollout_depth=1,  
        gamma=0.99  
    )

    
    best_action, _ = mcts.search(state, score)

    return best_action
