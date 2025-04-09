# run_voyager_omnigibson.py
import os
import sys
import config
from omnigibson_interface import OmniGibsonInterface
from omnigibson_env import OmniGibsonEnv
from llm_api import LLM_API

def run_agent():
    print("--- Starting Voyager-OmniGibson Run ---")

    # 1. Initialize Components
    try:
        print("Initializing LLM...")
        llm = LLM_API()
        print("Initializing Interface...")
        interface = OmniGibsonInterface()
        print("Initializing Env Wrapper...")
        env = OmniGibsonEnv(interface)
    except Exception as e:
        print(f"FATAL: Failed to initialize components: {e}")
        # Attempt cleanup if possible
        if 'env' in locals() and env: env.close()
        elif 'interface' in locals() and interface: interface.close()
        return

    # 2. Load Task and Prompt Template
    task_name = config.DEFAULT_TASK
    try:
        print(f"Resetting environment for task: {task_name}...")
        observation = env.reset(task_name)
        task_description = env.get_task_goal_description()
        prompt_template_path = os.path.join(config.PROMPT_DIR, config.ACTION_PROMPT_TEMPLATE_NAME)
        with open(prompt_template_path, 'r') as f:
            prompt_template = f.read()
        print("Task setup complete.")
    except FileNotFoundError as e:
        print(f"FATAL: Required file not found: {e}")
        env.close(); return
    except Exception as e:
        print(f"FATAL: Error during task setup: {e}")
        env.close(); return

    # 3. Agent Loop
    print(f"\n=== Starting Task: {task_name} ===")
    print(f"Goal: {task_description}")

    for step in range(config.MAX_STEPS_PER_TASK):
        print(f"\n--- Step {step + 1} / {config.MAX_STEPS_PER_TASK} ---")
        print(f"Current Observation:\n{observation}")

        # Construct prompt
        prompt = prompt_template.format(
            task_description=task_description,
            observation=observation,
        )

        # Get action from LLM
        action_code = llm.generate(prompt) # Uses max_new_tokens from config

        if not action_code or action_code.startswith("Error:"):
            print(f"WARN: LLM generation failed or returned error: {action_code}. Stopping task.")
            break # Stop if LLM fails

        # Execute action in environment
        observation, reward, done, info = env.step(action_code)

        print(f"Step Info: {info}")

        if done:
            print(f"\n=== Task '{task_name}' Succeeded in {step + 1} steps! ===")
            break
        elif step == config.MAX_STEPS_PER_TASK - 1:
            print(f"\n=== Task '{task_name}' Failed: Reached max steps ({config.MAX_STEPS_PER_TASK}) ===")
            break

    # 4. Cleanup
    print("Closing environment...")
    env.close()
    print("--- Run Finished ---")

if __name__ == "__main__":
    # Ensure running within the Apptainer context with headless flag
    print(f"Running script from: {os.getcwd()}")
    print(f"Python executable: {sys.executable}")
    if "OMNIGIBSON_HEADLESS" not in os.environ or os.environ["OMNIGIBSON_HEADLESS"] != "1":
         print("WARNING: OMNIGIBSON_HEADLESS=1 environment variable not set. Simulation might fail or try to render.")
         # sys.exit(1) # Optional: Force exit if not headless
    run_agent()
