import math
import random
from typing import List, Tuple, Union, Optional, Any
from agent import Agent
from connect_4_env import ConnectFourEnv
import math
import copy


class Node:
    def __init__(self, state: ConnectFourEnv, parent: Optional['Node'] = None, action: Optional[int] = None):
        self.state = copy.deepcopy(state)  # Copy the game state at this node
        self.parent = parent  # Parent node
        self.action = action  # Action that led to this state
        self.children: List['Node'] = []  # Child nodes
        self.wins: float = 0  # Total wins from this node
        self.visits: int = 0  # Total visits to this node
        # Untried actions from this state
        self.untried_actions: List[int] = self.state.get_legal_actions()

    def is_terminal(self) -> bool:
        # Check if the game is over (win, draw, or invalid move)
        return self.state.done

    def is_fully_expanded(self) -> bool:
        # A node is fully expanded if there are no untried actions left
        return len(self.untried_actions) == 0

    def add_child(self, action: int) -> 'Node':
        # Create a new child node for the given action
        new_state = copy.deepcopy(self.state)
        new_state.step(action)
        child_node = Node(new_state, self, action)
        self.untried_actions.remove(action)
        self.children.append(child_node)
        return child_node

    def all_moves_until_now(self) -> List[int]:
        # return the moves starting from the root node until this node
        moves = []
        current_node = self
        while current_node.parent is not None:
            moves.append(current_node.action)
            current_node = current_node.parent
        return moves[::-1]


class MCTS:
    def __init__(self, initial_state: ConnectFourEnv, c: float = 1.41):
        self.root = Node(initial_state)
        self.c = c  # Exploration parameter
        self.me = initial_state.current_player

    def select(self, node: Node) -> Node:
        while not node.is_terminal():
            if not node.is_fully_expanded():
                return self.expand(node)
            else:
                node = self.best_child(node)
        return node

    def expand(self, node: Node) -> Node:
        # Expand the node by adding a new child
        action = node.untried_actions[0]  # Select an untried action
        expanded_node = node.add_child(action)
        return expanded_node

    def simulate(self, node: Node) -> float:
        # Simulate a random play-out from the node's state
        simulation_env = copy.deepcopy(node.state)

        # check if the game is over

        if simulation_env.done:
            reward = 1 if simulation_env.winner == self.me else 0
        else:
            reward = 0

        while not simulation_env.done:
            possible_actions = simulation_env.get_legal_actions()
            action = random.choice(possible_actions)
            _, reward, _, _ = simulation_env.step(action)

        return reward

    def backpropagate(self, node: Node, result: float) -> None:
        # Update nodes with the simulation result up to the root

        while node is not None:
            node.visits += 1
            if node.state.current_player != self.me:
                node.wins += result
            else:
                node.wins += 1 - result
            node = node.parent

    def best_child(self, node: Node, eval=False) -> Node:
        # Select the child with the highest UCB1 score
        if eval:
            choices_weights = [
                (child.wins / child.visits) if child.visits > 0 else 0
                for child in node.children
            ]
        else:
            choices_weights = [
                (child.wins / child.visits) + self.c *
                math.sqrt((2 * math.log(node.visits) / child.visits))
                # Handle division by zero
                if child.visits > 0 else float('inf')
                for child in node.children
            ]

        return node.children[choices_weights.index(max(choices_weights))]

    def run(self, max_iterations: int) -> None:
        for _ in range(max_iterations):
            leaf = self.select(self.root)
            simulation_result = self.simulate(leaf)
            self.backpropagate(leaf, simulation_result)


class MonteCarloTreeSearchAgent(Agent):
    def __init__(self, env: ConnectFourEnv, n_iterations: int = 100, c: float = 1.41):
        super().__init__(env)
        self.n_iterations = n_iterations
        self.c = c

    def choose_action(self, verbose=False, select=True) -> int:
        # Run MCTS and choose the best action
        self.mcts = MCTS(copy.deepcopy(self.env), c=self.c)
        self.mcts.run(self.n_iterations)
        if verbose:
            print("The number of visits for each child node of the root:")
            print([n.visits for n in self.mcts.root.children])
            print("Average reward for each child node of the root:")
            print([n.wins / n.visits if n.visits >
                   0 else 0 for n in self.mcts.root.children])
        best_move = self.mcts.best_child(self.mcts.root, eval=True).action

        # if select and not self.mcts.root.is_terminal():
        #     new_root = self.mcts.best_child(self.mcts.root, eval=False)
        #     # Ensure the new root is not terminal and has children
        #     if new_root and not new_root.is_terminal() and new_root.children:
        #         self.mcts.root = new_root
        #     else:
        #         # Handle the case where a new non-terminal root cannot be found
        #         # This might include reinitializing MCTS with the current state
        #         # or handling the end of the game appropriately
        #         print(
        #             "No valid new root found. Game might be ending or requires handling.")

        return best_move


env = ConnectFourEnv()

# TEST ATTACK
# env.step(0)
# env.step(0)
# env.step(1)
# env.step(1)
# env.step(3)
# env.step(3)
# env.render()

# TEST DEFENSE
# env.step(0)
# env.step(1)
# env.step(2)
# env.step(1)
# env.step(3)
# env.step(1)
# env.render()

# monte = MonteCarloTreeSearchAgent(env, n_iterations=10000, c=1.4)
# print(monte.choose_action(verbose=True))


# env.play(monte, RandomAgent(env), n_games=1, show_game=True)
