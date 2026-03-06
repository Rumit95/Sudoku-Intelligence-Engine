import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib
matplotlib.use('TkAgg') # ✅ Set this explicitly before importing pyplot
import matplotlib.pyplot as plt
import os
import pandas as pd
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib import MaskablePPO


# ------------------------------------------------------------------
# ENVIRONMENT
# ------------------------------------------------------------------
class SudokuEnv(gym.Env):
    def __init__(self, df=None):
        super().__init__()

        self.df = df
        self.use_dataset = df is not None

        self.default_puzzle = np.array([
            [5,3,0, 0,7,0, 0,0,0],
            [6,0,0, 1,9,5, 0,0,0],
            [0,9,8, 0,0,0, 0,6,0],
            [8,0,0, 0,6,0, 0,0,3],
            [4,0,0, 8,0,3, 0,0,1],
            [7,0,0, 0,2,0, 0,0,6],
            [0,6,0, 0,0,0, 2,8,0],
            [0,0,0, 4,1,9, 0,0,5],
            [0,0,0, 0,8,0, 0,7,9],
        ])
        self.default_solution = np.array([
            [5,3,4, 6,7,8, 9,1,2],
            [6,7,2, 1,9,5, 3,4,8],
            [1,9,8, 3,4,2, 5,6,7],
            [8,5,9, 7,6,1, 4,2,3],
            [4,2,6, 8,5,3, 7,9,1],
            [7,1,3, 9,2,4, 8,5,6],
            [9,6,1, 5,3,7, 2,8,4],
            [2,8,7, 4,1,9, 6,3,5],
            [3,4,5, 2,8,6, 1,7,9],
        ])

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(9, 9, 9), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2 * 9 * 9 * 9)

        self.board        = None
        self.candidates   = None
        self.given_mask   = None
        self.state        = None
        self.total_reward = 0.0
        self.puzzle       = self.default_puzzle
        self.solution     = self.default_solution
        self.failed_actions = set() # ✅ Track actions that failed in the current episode

    # ------------------------------------------------------------------
    def _decode_action(self, action):
        level = action // (9 * 9 * 9)
        rem   = action  % (9 * 9 * 9)
        row   = rem // (9 * 9)
        rem   = rem  % (9 * 9)
        col   = rem // 9
        dig   = rem  % 9
        return level, row, col, dig

    def _is_valid_candidate(self, row, col, digit_1indexed):
        d = digit_1indexed
        if d in self.board[row, :]:   return False
        if d in self.board[:, col]:   return False
        br, bc = (row // 3) * 3, (col // 3) * 3
        if d in self.board[br:br+3, bc:bc+3]: return False
        return True

    def _build_state(self):
        self.state = self.candidates.astype(np.float32)

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # ✅ Check for custom puzzle in options
        if options and "puzzle" in options:
            self.puzzle = np.array(list(options["puzzle"]), dtype=int).reshape(9, 9)
            if "solution" in options:
                self.solution = np.array(list(options["solution"]), dtype=int).reshape(9, 9)
            else:
                # If no solution provided, we use a dummy one (won't be able to provide +1.0 rewards)
                self.solution = np.zeros((9, 9), dtype=int)
        # ✅ pick random puzzle each episode
        elif self.use_dataset:
            idx          = np.random.randint(len(self.df))
            puzzle_str   = self.df.iloc[idx]['quizzes']
            solution_str = self.df.iloc[idx]['solutions']
            self.puzzle   = np.array(list(puzzle_str),   dtype=int).reshape(9, 9)
            self.solution = np.array(list(solution_str), dtype=int).reshape(9, 9)
        else:
            self.puzzle   = self.default_puzzle
            self.solution = self.default_solution

        self.board      = self.puzzle.copy().astype(int)
        self.given_mask = (self.puzzle > 0)
        self.total_reward = 0.0
        self.failed_actions.clear() # ✅ Reset memory for the new episode

        self.candidates = np.zeros((9, 9, 9), dtype=bool)
        for i in range(9):
            for j in range(9):
                if self.given_mask[i, j]:
                    d = self.puzzle[i, j] - 1
                    self.candidates[i, j, d] = True

        self._build_state()
        return self.state.copy(), {}

    # ------------------------------------------------------------------
    def step(self, action):
        action = int(action) # ✅ Convert numpy scalar to Python int for hashability
        level, row, col, dig = self._decode_action(action)
        digit_1indexed = dig + 1
        reward = 0.0
        info   = {}

        if self.given_mask[row, col]:
            reward = -0.3
            info["reason"] = "tried to modify a given cell"
            self._build_state()
            return self.state.copy(), reward, self._check_done(), False, info

        if level == 0:
            if self.candidates[row, col, dig]:
                reward = -0.1
                info["reason"] = "candidate already added"
            elif self._is_valid_candidate(row, col, digit_1indexed):
                self.candidates[row, col, dig] = True
                reward = +0.5
                info["reason"] = "valid candidate added"
            else:
                reward = -0.5
                info["reason"] = "invalid candidate"

        elif level == 1:
            if self.board[row, col] != 0:
                reward = -0.3
                info["reason"] = "cell already filled"
            elif not self.candidates[row, col, dig]:
                reward = -0.5
                info["reason"] = "candidate not added first"
            elif digit_1indexed == self.solution[row, col]:
                self.board[row, col] = digit_1indexed
                self.candidates[row, col, :] = False
                self.candidates[row, col, dig] = True
                reward = +1.0
                info["reason"] = "correct placement"
            else:
                reward = -1.0
                info["reason"] = "wrong digit"

        if reward < 0:
            self.failed_actions.add(action) # ✅ Remember this action failed

        self.total_reward += reward
        self._build_state()
        done = self._check_done()

        # ✅ bonus reward for solving the full puzzle
        if done:
            reward += 10.0
            self.total_reward += 10.0
            info["reason"] = "🎉 puzzle solved!"

        return self.state.copy(), reward, done, False, info

    def action_masks(self):
        mask = np.zeros(2 * 9 * 9 * 9, dtype=bool)
        
        for act in range(len(mask)):
            # ✅ Skip actions we already know will fail (act is already an int here)
            if act in self.failed_actions:
                continue
                
            level, row, col, dig = self._decode_action(act)
            
            # 1. Never allow modifying given cells
            if self.given_mask[row, col]:
                continue
                
            if level == 0: # Pencil/Candidate
                # Invalid if candidate already exists
                if self.candidates[row, col, dig]:
                    continue
                # Invalid if Sudoku rules broken
                if not self._is_valid_candidate(row, col, dig + 1):
                    continue
                mask[act] = True
                
            elif level == 1: # Placement
                # Invalid if cell already filled
                if self.board[row, col] != 0:
                    continue
                # Invalid if candidate not yet placed (pencil-mark-first rule)
                if not self.candidates[row, col, dig]:
                    continue
                # In this specific env, Level 1 also checks if dig is correct solution
                # To make it slightly easier but not cheating, we block obviously wrong moves
                mask[act] = True
                
        return mask

    def _check_done(self):
        return all(
            self.board[i, j] != 0
            for i in range(9) for j in range(9)
            if not self.given_mask[i, j]
        )

    # ------------------------------------------------------------------
    # def render(self, save_path=None):
    #     if self.state is None:
    #         return

    #     fig, ax = plt.subplots(figsize=(7, 7))
    #     fig.patch.set_facecolor('white')
    #     ax.set_facecolor('white')

    #     bg = np.ones((9, 9))
    #     for i in range(9):
    #         for j in range(9):
    #             if self.given_mask[i, j]:
    #                 bg[i, j] = 0.75

    #     ax.imshow(bg, cmap='gray', vmin=0, vmax=1,
    #               extent=[0, 9, 0, 9], aspect='equal',
    #               zorder=0, interpolation='none')

    #     for i in range(10):
    #         lw = 2.5 if i % 3 == 0 else 0.5
    #         ax.plot([0, 9], [i, i], 'k', linewidth=lw, zorder=1)
    #         ax.plot([i, i], [0, 9], 'k', linewidth=lw, zorder=1)

    #     for i in range(9):
    #         for j in range(9):
    #             cell = self.state[i, j]
    #             if self.board[i, j] != 0:
    #                 color = 'black' if self.given_mask[i, j] else 'blue'
    #                 ax.text(j + 0.5, 9 - (i + 0.5), str(self.board[i, j]),
    #                         ha='center', va='center',
    #                         fontsize=18, fontweight='bold', color=color, zorder=2)
    #             else:
    #                 for num in range(9):
    #                     if cell[num] > 0.5:
    #                         sub_row, sub_col = num // 3, num % 3
    #                         x = j + (sub_col + 0.5) / 3
    #                         y = 9 - (i + (sub_row + 0.5) / 3)
    #                         ax.text(x, y, str(num + 1),
    #                                 ha='center', va='center',
    #                                 fontsize=6, color='green', zorder=2)

    #     ax.set_xlim(0, 9)
    #     ax.set_ylim(0, 9)
    #     ax.axis('off')
    #     ax.set_title(f"Total Reward: {self.total_reward:.1f}", fontsize=12)

    #     path = save_path or "renders/sudoku_latest.png"
    #     fig.savefig(path, bbox_inches='tight', dpi=100)
    #     plt.close(fig)

    def render(self):
        if self.state is None:
            return

        if not hasattr(self, "fig"):
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(7, 7))
            self.fig.patch.set_facecolor('white')  # ✅ figure background white

        self.ax.clear()
        self.ax.set_facecolor('white')             # ✅ axes background white

        # ✅ imshow — 1.0=white (empty cells), 0.75=grey (given cells)
        bg = np.ones((9, 9))                       # start all white
        for i in range(9):
            for j in range(9):
                if self.given_mask[i, j]:
                    bg[i, j] = 0.75               # grey for given cells

        self.ax.imshow(
            bg, cmap='gray', vmin=0, vmax=1,
            extent=[0, 9, 0, 9], aspect='equal',
            zorder=0, interpolation='none'
        )

        # Grid lines
        for i in range(10):
            lw = 2.5 if i % 3 == 0 else 0.5
            self.ax.plot([0, 9], [i, i], 'k', linewidth=lw, zorder=1)
            self.ax.plot([i, i], [0, 9], 'k', linewidth=lw, zorder=1)

        for i in range(9):
            for j in range(9):
                cell = self.state[i, j]

                if self.board[i, j] != 0:
                    color = 'black' if self.given_mask[i, j] else 'blue'
                    self.ax.text(
                        j + 0.5, 9 - (i + 0.5),
                        str(self.board[i, j]),
                        ha='center', va='center',
                        fontsize=18, fontweight='bold', color=color, zorder=2
                    )
                else:
                    for num in range(9):
                        if cell[num] > 0.5:
                            sub_row = num // 3
                            sub_col = num % 3
                            x = j + (sub_col + 0.5) / 3
                            y = 9 - (i + (sub_row + 0.5) / 3)
                            self.ax.text(
                                x, y, str(num + 1),
                                ha='center', va='center',
                                fontsize=6, color='green', zorder=2
                            )

        self.ax.set_xlim(0, 9)
        self.ax.set_ylim(0, 9)
        self.ax.axis('off')
        self.ax.set_title(f"Total Reward: {self.total_reward:.1f}", fontsize=12)

        self.fig.canvas.draw_idle() # ✅ Recommended for better stability
        self.fig.canvas.flush_events()
        plt.pause(0.001) 

    def close_render(self):
        if hasattr(self, "fig"):
            plt.close(self.fig)
            del self.fig
            del self.ax


# ------------------------------------------------------------------
# TRAINING CALLBACK — logs progress
# ------------------------------------------------------------------
class SudokuCallback(BaseCallback):
    def __init__(self, eval_env, eval_every=10_000, verbose=1):
        super().__init__(verbose)
        self.eval_env    = eval_env
        self.eval_every  = eval_every
        self.rewards     = []
        self.solve_rates = []
        self.best_reward = -np.inf
        plt.ion() # ✅ Enable interactive mode immediately

    def _on_step(self):
        # ✅ Log every 1000 steps so we know it's alive
        if self.num_timesteps % 1000 < self.model.n_envs:
            print(f"   [Training] Step {self.num_timesteps:,}...")

        # ✅ Trigger based on total timesteps (matches progress bar)
        if self.num_timesteps % self.eval_every < self.model.n_envs:
            solved      = 0
            total_rew   = 0
            n_eval      = 5        # reduced for faster preview during training

            for ep_idx in range(n_eval):
                obs, _ = self.eval_env.reset()
                ep_rew = 0
                for step_idx in range(2000):          # max steps per episode
                    # ✅ Get masks for the model
                    masks = self.eval_env.action_masks()
                    action, _ = self.model.predict(obs, action_masks=masks, deterministic=True)
                    obs, rew, done, _, info = self.eval_env.step(action)
                    
                    if ep_idx == 0: 
                        self.eval_env.render()
                        # ✅ Print what the model is actually doing
                        if step_idx < 10: # Only print first 10 steps to avoid spam
                            print(f"   Step {step_idx}: Action {action} -> {info.get('reason')}")
                        elif step_idx == 10:
                            print("   ...")
                        
                    ep_rew += rew
                    if done:
                        solved += 1
                        break
                    
                # ✅ Close the visualization window after the first episode
                if ep_idx == 0:
                    self.eval_env.unwrapped.close_render()

                total_rew += ep_rew

            avg_rew   = total_rew / n_eval
            solve_rate = solved / n_eval * 100

            self.rewards.append(avg_rew)
            self.solve_rates.append(solve_rate)

            print(f"\n📊 Step {self.n_calls:,} | "
                  f"Avg Reward: {avg_rew:.1f} | "
                  f"Solve Rate: {solve_rate:.0f}% ({solved}/{n_eval})")

            # ✅ save best model
            if avg_rew > self.best_reward:
                self.best_reward = avg_rew
                self.model.save("models/best_sudoku_model")
                print(f"   💾 New best model saved! (reward={avg_rew:.1f})")

            # ✅ plot training curve
            self._plot_progress()

        return True

    def _plot_progress(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(self.rewards, 'b-o', markersize=3)
        ax1.set_title('Average Reward per Eval')
        ax1.set_xlabel('Eval #')
        ax1.set_ylabel('Avg Reward')
        ax1.grid(True)

        ax2.plot(self.solve_rates, 'g-o', markersize=3)
        ax2.set_title('Solve Rate %')
        ax2.set_xlabel('Eval #')
        ax2.set_ylabel('Solve Rate (%)')
        ax2.set_ylim(0, 100)
        ax2.grid(True)

        plt.tight_layout()
        fig.savefig('renders/training_progress.png', dpi=100)
        plt.close(fig)


# ------------------------------------------------------------------
# MAIN — TRAIN
# ------------------------------------------------------------------
if __name__ == "__main__":

    os.makedirs("renders", exist_ok=True)
    os.makedirs("models",  exist_ok=True)

    full_df = pd.read_csv("sudoku 1 mil/sudoku.csv")       # ✅ put your dataset here
    # download: https://www.kaggle.com/datasets/bryanpark/sudoku
    print(f"✅ Loaded {len(full_df)} puzzles from sudoku 1 mil/sudoku.csv")

    # ✅ training environment (4 parallel envs)
    def make_masked_env():
        env = SudokuEnv(df=full_df)
        env = ActionMasker(env, lambda e: e.action_masks()) # ✅ Add ActionMasker
        return Monitor(env)

    train_env = make_vec_env(make_masked_env, n_envs=2, seed=42)

    # ✅ separate eval environment
    eval_env = SudokuEnv(df=full_df)
    eval_env = ActionMasker(eval_env, lambda e: e.action_masks())

    # ✅ MaskablePPO model
    model = MaskablePPO(
        "MlpPolicy",
        train_env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        clip_range=0.2,
        tensorboard_log="./tensorboard_logs/",
    )

    # ✅ Better policy for Sudoku:
    # model = PPO(
    #     "CnnPolicy",          # understands spatial structure
    #     train_env,
    #     policy_kwargs=dict(
    #         features_extractor_kwargs=dict(features_dim=256)
    #     ),
    #     ...
    # )

    print("=" * 50)
    print("🚀 Starting Training")
    print("=" * 50)
    print(f"  Algorithm     : PPO")
    print(f"  Action space  : {eval_env.action_space}")
    print(f"  Obs shape     : {eval_env.observation_space.shape}")
    print(f"  Total steps   : 1,000,000")
    print("=" * 50 + "\n")

    callback = SudokuCallback(eval_env=eval_env, eval_every=10_000)

    # ✅ train
    model.learn(
        total_timesteps=1_000_000,
        callback=callback,
        progress_bar=False # ✅ Set to False to prevent 'rich' threading crash
    )

    # ✅ save final model
    model.save("models/final_sudoku_model")
    print("\n✅ Training complete!")
    print("   Models saved in models/")
    print("   Training curve saved in renders/training_progress.png")

    # ------------------------------------------------------------------
    # TEST TRAINED MODEL
    # ------------------------------------------------------------------
    # print("\n" + "=" * 50)
    # print("🧪 Testing trained model")
    # print("=" * 50)

    # model = PPO.load("models/best_sudoku_model")
    # test_env = SudokuEnv(csv_path=CSV_PATH)

    # solved = 0
    # n_test = 50

    # for episode in range(n_test):
    #     obs, _ = test_env.reset()
    #     test_env.render(save_path=f"renders/test_start_{episode:02d}.png")

    #     for step in range(2000):
    #         action, _ = model.predict(obs, deterministic=True)
    #         obs, reward, done, _, info = test_env.step(action)
    #         if done:
    #             solved += 1
    #             test_env.render(save_path=f"renders/test_solved_{episode:02d}.png")
    #             print(f"  Episode {episode+1:2d} ✅ solved in {step+1} steps | "
    #                   f"reward={test_env.total_reward:.1f}")
    #             break
    #     else:
    #         print(f"  Episode {episode+1:2d} ❌ not solved | "
    #               f"reward={test_env.total_reward:.1f}")

    # print(f"\n🏆 Final solve rate: {solved}/{n_test} = {solved/n_test*100:.0f}%")