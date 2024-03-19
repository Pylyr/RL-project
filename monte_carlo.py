import math
import random
import copy
from typing import List, Tuple, Union, Optional, Any
from env import ConnectFourEnv
import math
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def _hierarchy_pos(G, root, width=1.0, vert_gap=0.2, xcenter=0.5, pos=None, level=0):
    """
    Create a layout for a tree where the root is at the top and children are placed below parents
    """
    if pos is None:
        pos = {root: (xcenter, 1 - level * vert_gap)}
    else:
        pos[root] = (xcenter, 1 - level * vert_gap)

    children = list(G.successors(root))
    if len(children) != 0:
        dx = width / len(children)
        nextx = xcenter - width / 2 - dx / 2
        for child in children:
            nextx += dx
            pos = _hierarchy_pos(
                G, child, width=dx, vert_gap=vert_gap, xcenter=nextx, pos=pos, level=level + 1)
    return pos


def hierarchy_pos(G, root, width=1., vert_gap=0.2, xcenter=0.5):
    """
    Create a layout for a tree displayed with its root at the top.

    Adapted from Joel's answer at https://stackoverflow.com/a/29597209/2966723.
    """
    pos = _hierarchy_pos(G, root, width, vert_gap, xcenter)
    return pos


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


class MCTS:
    def __init__(self, initial_state: ConnectFourEnv, c: float = 1.41):
        self.root = Node(initial_state)
        self.c = c  # Exploration parameter
        self.graph = nx.DiGraph()

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
        return node.add_child(action)

    def simulate(self, node: Node) -> float:
        # Simulate a random play-out from the node's state
        simulation_env = copy.deepcopy(node.state)
        while not simulation_env.done:
            possible_actions = simulation_env.get_legal_actions()
            action = random.choice(possible_actions)
            simulation_env.step(action)
        return simulation_env.rewards['win'] if simulation_env.winner == node.state.current_player else 0

    def backpropagate(self, node: Node, result: float) -> None:
        # Update nodes with the simulation result up to the root
        while node is not None:
            node.visits += 1
            if node.parent and node.state.current_player != node.parent.state.current_player:
                node.wins += result  # Only update win count if the node's player is the winner
            node = node.parent

    def best_child(self, node: Node) -> Node:
        # Select the child with the highest UCB1 score
        choices_weights = [
            (child.wins / child.visits) + self.c *
            math.sqrt((2 * math.log(node.visits) / child.visits))
            if child.visits > 0 else float('inf')  # Handle division by zero
            for child in node.children
        ]
        return node.children[choices_weights.index(max(choices_weights))]

    def run(self, max_iterations: int) -> None:
        for _ in range(max_iterations):
            leaf = self.select(self.root)
            simulation_result = self.simulate(leaf)
            self.backpropagate(leaf, simulation_result)

    def node_from_name(self, name: str) -> Optional[Node]:
        """
        Given a node name, retrieve the corresponding Node instance.
        """
        # Recursive helper function to search for the node
        def search_node(current_node: Node, name: str) -> Optional[Node]:
            if f"{current_node.action}-{current_node.wins}/{current_node.visits}" == name:
                return current_node
            for child in current_node.children:
                result = search_node(child, name)
                if result:
                    return result
            return None

        # If name is 'root', return the root node
        if name == 'root':
            return self.root

        return search_node(self.root, name)

    def render(self):
        self._build_graph(self.root)
        # draw the graph

        pos = hierarchy_pos(self.graph, 'root')

        fig, ax = plt.subplots(figsize=(10, 10))
        nx.draw(self.graph, pos, with_labels=False,
                node_size=100, arrows=True)

        # Draw images on the nodes
        ax = plt.gca()
        for node, (x, y) in pos.items():
            if node != 'root':  # Skip the root for simplicity
                # You need to implement this method
                state = self.node_from_name(node).state
                img = state.to_image()
                bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                imsize = 2  # size relative to the figure
                ximg = x - imsize/2.0
                yimg = y - imsize/2.0
                imgbox = OffsetImage(img, zoom=imsize/bbox.width)
                ab = AnnotationBbox(imgbox, (x, y), frameon=False, pad=0)
                ax.add_artist(ab)

        plt.show()

    def _build_graph(self, node: Node, parent_name: Optional[str] = None):
        # Create a unique name for the node based on its state and action
        node_name = f"{node.action}-{node.wins}/{node.visits}" if node.action is not None else "root"
        self.graph.add_node(node_name)

        # If this node has a parent, add an edge from the parent to this node
        if parent_name is not None and node.action is not None:
            self.graph.add_edge(parent_name, node_name,
                                action=str(node.action))

        # Recursively add child nodes and edges to the graph
        for child in node.children:
            self._build_graph(child, node_name)


class MonteCarloTreeSearchAgent:
    def __init__(self, env: ConnectFourEnv, n_iterations: int = 100, c: float = 1.41):
        self.env = env
        self.n_iterations = n_iterations
        self.c = c

    def choose_action(self) -> int:
        # Run MCTS and choose the best action
        mcts = MCTS(copy.deepcopy(self.env), c=self.c)
        mcts.run(self.n_iterations)
        mcts.render()
        best_move = mcts.best_child(mcts.root).action
        return best_move


env = ConnectFourEnv()
monte = MonteCarloTreeSearchAgent(env, n_iterations=40)
print(monte.choose_action())
