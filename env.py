import gym
from gym import spaces
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
from PIL import Image


class ConnectFourEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    rewards = {
        'win': 1000,
        'draw': 500,
        'nothing': -10,
        'invalid': -1000,
    }

    def __init__(self):
        super(ConnectFourEnv, self).__init__()
        self.rows = 6
        self.columns = 7
        self.in_a_row = 4
        self.action_space = spaces.Discrete(self.columns)
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(self.rows, self.columns), dtype=int)
        self.winner = 0
        self.reset()

    def flip_board(self):
        self.board = [[3 - x if x != 0 else 0 for x in row]
                      for row in self.board]

    def step(self, action):
        # Check if action is valid
        if self.playable_rows[action] == -1:
            self.winner = 3 - self.current_player
            return self.board, self.rewards['invalid'], True, {}

        if self.done:
            raise ValueError("Game is over, please reset the environment")

        # Find the next open row
        playable_row = self.playable_rows[action]
        self.board[playable_row][action] = self.current_player
        self.playable_rows[action] -= 1

        reward, self.done = self.check_win(playable_row, action)
        self.current_player = 3 - self.current_player  # Switch player
        return self.board, reward, self.done, {}

    def reset(self):
        self.board = [[0] * self.columns for _ in range(self.rows)]
        self.playable_rows = [self.rows - 1] * self.columns
        self.winner = 0
        self.current_player = 1
        self.done = False
        return self.board

    def _render(self, mode='human'):
        fig, ax = plt.subplots(figsize=(2, 2), dpi=80)  # Smaller figure size
        # Draw the board
        for x in range(self.columns):
            for y in range(self.rows):
                circle = patches.Circle(
                    (x, self.rows - y - 1), 0.45, fill=False, color='black', lw=2)
                if self.board[y][x] == 1:
                    circle = patches.Circle(
                        (x, self.rows - y - 1), 0.45, fill=True, color='red', lw=2)
                elif self.board[y][x] == 2:
                    circle = patches.Circle(
                        (x, self.rows - y - 1), 0.45, fill=True, color='yellow', lw=2)
                ax.add_patch(circle)
        plt.xlim(-0.5, self.columns-0.5)
        plt.ylim(-0.5, self.rows-0.5)
        ax.set_aspect('equal')
        plt.grid()
        plt.xticks(range(self.columns))
        plt.yticks(range(self.rows))
        plt.axis('off')
        return fig

    def render(self, mode='human'):
        self._render(mode)
        plt.show()

    def to_image(self, save_path=None):
        fig = self._render()
        # Save the figure to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)

        # Convert buffer to a PIL Image
        img = Image.open(buf)
        if save_path:
            img.save(save_path)
        # buf.close()
        return img

    def check_win(self, row, col):
        # Check only from the last move's position
        # Horizontal, vertical, diagonal down, diagonal up
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dx, dy in directions:
            count = 1  # Count includes the last move
            # Check in the positive direction
            for i in range(1, 4):
                x, y = row + dx*i, col + dy*i
                if x < 0 or x >= self.rows or y < 0 or y >= self.columns or self.board[x][y] != self.current_player:
                    break
                count += 1
            # Check in the negative direction
            for i in range(1, 4):
                x, y = row - dx*i, col - dy*i
                if x < 0 or x >= self.rows or y < 0 or y >= self.columns or self.board[x][y] != self.current_player:
                    break
                count += 1
            if count >= self.in_a_row:
                self.winner = self.current_player
                return self.rewards['win'], True

        # Check for draw
        if all([x == self.rows for x in self.playable_rows]):
            self.winner = 3 - self.current_player
            return self.rewards['draw'], True
        return self.rewards['nothing'], False

    def check_win_globally(self):
        for row in range(self.rows):
            for col in range(self.columns):
                if self.board[row][col] != 0:
                    _, done = self.check_win(row, col)
                    if done:
                        return True
        return False

    def load_board(self, board):
        self.board = board
        self.playable_rows = [self.rows - 1 - sum([1 for j in range(self.columns)
                                                   if board[i][j] != 0]) for i in range(self.rows)]
        self.current_player = 1 if sum(
            [1 for row in board for cell in row if cell != 0]) % 2 == 0 else 2
        self.done = self.check_win_globally()

    def play(self, agent1, agent2, n_games=1000, show_game=False, show_outcome=False, start_board=None):
        # Play n games between two agents
        results = {1: 0, 2: 0}
        game_lengths = []
        for _ in range(n_games):
            if start_board is None:
                self.reset()
            else:
                self.load_board(start_board)
            done = False
            while not done:
                if show_game:
                    self.render()

                action = agent1.choose_action()
                _, _, done, _ = self.step(action)
                if done:
                    break

                if show_game:
                    self.render()

                action = agent2.choose_action()
                _, _, done, _ = self.step(action)

            results[self.winner] += 1
            game_lengths.append(
                self.rows * (self.columns - 1) - sum(self.playable_rows))

            if show_outcome or show_game:
                self.render()

        avg_game_length = sum(game_lengths) / len(game_lengths)
        return results, avg_game_length

    def get_legal_actions(self):
        return [col for col in range(self.columns) if self.playable_rows[col] != -1]
