# omnigibson_env.py
import re
import ast # For safely evaluating literals if needed, but avoid exec/eval directly

import config
from omnigibson_interface import OmniGibsonInterface

class OmniGibsonEnv:
    def __init__(self, interface: OmniGibsonInterface):
        print("Initializing Voyager Environment Wrapper...")
        self.interface = interface
        self.action_list = [
            "navigate_to_object(object_name: str)",
            "toggle_object(object_name: str)",
            "pick_up_object(object_name: str)",
            "place_object_on(object_to_place: str, surface_object_name: str)",
        ]
        self.available_actions_description = "\n- ".join(self.action_list)
        print("Voyager Environment Wrapper Initialized.")

    def get_available_actions(self):
        """Returns a string describing available actions for the prompt."""
        return self.available_actions_description

    def reset(self, task_name=config.DEFAULT_TASK):
        """Resets the environment using the interface and returns the initial observation."""
        print(f"Resetting environment for task: {task_name}")
        initial_obs = self.interface.load_task(task_name)
        return initial_obs

    def _parse_action_code(self, action_code: str):
        """
        Parses the LLM's Python code string into function name and arguments.
        Uses regex for a simple 'function_name("arg1", "arg2")' format.
        WARNING: This is a basic parser and might fail on complex or malformed outputs.
                 A more robust parser (e.g., using ast) might be needed depending
                 on the LLM's output consistency. Avoid using exec() or eval() directly!
        """
        # Basic regex to capture function_name('arg1', ...) or function_name("arg1", ...)
        # It expects string literals as arguments for simplicity now.
        match = re.match(r"(\w+)\s*\(\s*(?:['\"]([^'\"]*)['\"](?:,\s*['\"]([^'\"]*)['\"])?)?\s*\)", action_code.strip())

        if match:
            function_name = match.group(1)
            args = [arg for arg in match.groups()[1:] if arg is not None] # Collect captured arguments
            print(f"Parsed action: {function_name}, Args: {args}")
            return function_name, args
        else:
            print(f"WARN: Could not parse action code: '{action_code}'")
            return None, None # Indicate parsing failure


    def step(self, action_code: str):
        """
        Takes raw action code from LLM, parses it, executes via interface,
        steps simulation, gets observation, and checks success.
        """
        print(f"\n--- Step: Received Action Code ---")
        print(action_code)
        print("---------------------------------")

        function_name, args = self._parse_action_code(action_code)

        action_success = False
        action_message = "Failed to parse action."

        if function_name:
            action_success, action_message = self.interface.execute_action(function_name, args)
            print(f"Action Execution Status: Success={action_success}, Msg='{action_message}'")
        else:
             # Handle parsing failure - maybe penalize the agent or just report failure
             pass

        # Step simulation regardless of action success to advance time
        raw_obs_dict = self.interface.step_simulation()

        # Get formatted observation for the next step
        observation = self.interface.get_observation(raw_obs_dict)

        # Check if the task goal is met *after* this step
        done = self.interface.check_success()

        # Simple reward: 1 if successful, 0 otherwise
        # Could be more complex (e.g., penalties for errors, rewards for progress)
        reward = 1.0 if done else 0.0

        # Info dictionary can contain debugging info
        info = {
            'action_code': action_code,
            'parsed_function': function_name,
            'parsed_args': args,
            'action_success': action_success,
            'action_message': action_message,
        }

        return observation, reward, done, info

    def close(self):
        """Closes the underlying environment interface."""
        self.interface.close()

    def get_task_goal_description(self):
        """Gets the task goal description from the interface."""
        return self.interface.get_task_goal_description()
