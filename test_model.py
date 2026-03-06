import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

# Import the environment and helper from the main script
from logical_3 import SudokuEnv

def test_on_random_puzzles(model_path="models/best_sudoku_model", csv_path="sudoku 1 mil/sudoku.csv", n_episodes=5, custom_str=None):
    """
    Loads a trained MaskablePPO model and tests it on random puzzles from the dataset or a custom string.
    """
    print(f"\n" + "=" * 50)
    print(f"🧪 TESTING MODEL: {model_path}")
    if custom_str:
        print(f"🧩 Custom Quiz: {custom_str}")
    print("=" * 50)

    # 1. Load the dataset (only if needed)
    df = None
    if not custom_str:
        if not os.path.exists(csv_path):
            print(f"❌ Error: Could not find dataset at {csv_path}")
            return
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df)} puzzles for testing.")

    # 2. Create and wrap the environment
    env = SudokuEnv(df=df)
    env = ActionMasker(env, lambda e: e.action_masks())
    
    # 3. Load the model
    if not os.path.exists(model_path + ".zip") and not os.path.exists(model_path):
        print(f"❌ Error: Could not find model at {model_path}")
        return

    model = MaskablePPO.load(model_path)
    print(f"✅ Model loaded successfully.")

    # 4. Testing loop
    effective_n = 1 if custom_str else n_episodes
    for episode in range(effective_n):
        if custom_str:
            obs, _ = env.reset(options={"puzzle": custom_str})
        else:
            obs, _ = env.reset()
        
        done = False
        step_count = 0
        
        if not custom_str:
            print(f"\n🚀 Episode {episode + 1}/{n_episodes} starting...")
        else:
            print(f"\n🚀 Testing custom puzzle...")
        
        while not done:
            # Get the action mask for the current state
            # Note: evaluate_policy or direct model.predict with masks
            masks = env.unwrapped.action_masks()
            
            # Predict the next move
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            
            # Step the environment
            obs, reward, done, truncated, info = env.step(action)
            
            # Render the board live
            env.unwrapped.render()
            
            # Log the action (truncated to avoid spam)
            if step_count < 15:
                print(f"   Step {step_count:2d}: Action {action:4d} -> {info.get('reason')}")
            elif step_count == 15:
                print("   ...")
            
            step_count += 1
            if step_count > 10000: # Safety break
                print("   🛑 Episode timed out (too many steps).")
                break
        
        if done:
            print(f"✅ Episode {episode + 1} SOLVED in {step_count} steps! Total Reward: {env.unwrapped.total_reward:.1f}")
        
        # Wait a moment before closing window and moving to next puzzle
        plt.pause(2.0)
        env.unwrapped.close_render()

    print("\n" + "=" * 50)
    print("✨ Testing complete!")
    print("=" * 50)

if __name__ == "__main__":
    # --- CUSTOM PUZZLE TEST ---
    # To test a custom puzzle, paste the 81-digit string here:
    # my_custom_sudoku = "000000000000003085001020000000507000004000100090000000500000073002010000000040009" 
    my_custom_sudoku = ""
    # my_custom_sudoku = "000000000000000000000000000000000000000000000000000000000000000000000000000000000"
    
    # If my_custom_sudoku is empty, it will test random puzzles from the dataset instead.
    test_on_random_puzzles(
        model_path="models/best_sudoku_model_kgl_9", 
        csv_path="sudoku 1 mil/sudoku.csv",
        n_episodes=3,
        custom_str=my_custom_sudoku if my_custom_sudoku else None
    )
