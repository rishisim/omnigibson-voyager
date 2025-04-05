# run_voyager_omnigibson.py
import os
import config
from omnigibson_interface import OmniGibsonInterface
from omnigibson_env import OmniGibsonEnv
from llm_api import LLM_API

def run_agent():
    print("--- Starting Voyager-OmniGibson Run ---")

    # 1. Initialize Components
    try:
        llm = LLM_API()
        interface = OmniGibsonInterface()
        env = OmniGibsonEnv(interface)
    except Exception as e:
        print(f"FATAL: Failed to initialize components: {e}")
        return # Cannot proceed

    # 2. Load Task and Prompt Template
    task_name = config.DEFAULT_TASK
    try:
        observation = env.reset(task_name)
        task_description = env.get_task_goal_description()
        prompt_template_path = os.path.join(config.PROMPT_DIR, config.ACTION_PROMPT_TEMPLATE_NAME)
        with open(prompt_template_path, 'r') as f:
            prompt_template = f.read()
    except FileNotFoundError as e:
        print(f"FATAL: Required file not found: {e}")
        env.close()
        return
    except Exception as e:
        print(f"FATAL: Error during setup: {e}")
        env.close()
        return


    # 3. Agent Loop
    print(f"\n=== Starting Task: {task_name} ===")
    print(f"Goal: {task_description}")

    for step in range(config.MAX_STEPS_PER_TASK):
        print(f"\n--- Step {step + 1} / {config.MAX_STEPS_PER_TASK} ---")
        print(f"Current Observation:\n{observation}")

        # Construct prompt for LLM
        prompt = prompt_template.format(
            task_description=task_description,
            observation=observation,
            # available_actions=env.get_available_actions() # Included in template text
        )

        # Get action from LLM
        action_code = llm.generate(prompt, max_new_tokens=50) # Adjust max_tokens as needed

        if not action_code or "Error:" in action_code:
            print("WARN: LLM failed to generate a valid action or returned an error. Skipping step.")
            # Optionally: Implement retry or stop logic
            # For now, just try stepping the sim with no action
            observation, reward, done, info = env.step("") # Empty action string
        else:
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
    # Ensure running within the Apptainer context where OmniGibson is available
    run_agent()
