# omnigibson_env.py
import re
import config
from omnigibson_interface import OmniGibsonInterface

class OmniGibsonEnv:
    def __init__(self, interface: OmniGibsonInterface):
        print("Initializing Voyager Environment Wrapper...")
        self.interface = interface
        # Define available actions for the prompt and parsing
        self.action_list = [
            "navigate_to_object(object_name: str)",
            "toggle_object(object_name: str)",
            "pick_up_object(object_name: str)",
            "place_object_on(object_to_place: str, surface_object_name: str)",
        ]
        self.available_actions_description = "\n- ".join(self.action_list)
        self.valid_action_names = {name.split('(')[0] for name in self.action_list}
        print("Voyager Environment Wrapper Initialized.")

    def get_available_actions(self):
        return self.available_actions_description

    def reset(self, task_name=config.DEFAULT_TASK):
        print(f"Resetting environment for task: {task_name}")
        initial_obs = self.interface.load_task(task_name)
        return initial_obs

    def _parse_action_code(self, action_code: str):
        """Parses LLM Python code string: function_name('arg1', 'arg2')"""
        action_code = action_code.strip().replace("'", '"') # Standardize quotes
        match = re.match(r"(\w+)\s*\(\s*(?:\"([^\"]*)\"(?:,\s*\"([^\"]*)\")?)?\s*\)", action_code)

        if match:
            function_name = match.group(1)
            if function_name not in self.valid_action_names:
                 print(f"WARN: Parsed function '{function_name}' is not in the valid action list.")
                 return None, None
            args = [arg for arg in match.groups()[1:] if arg is not None]
            print(f"Parsed action: {function_name}, Args: {args}")
            return function_name, args
        else:
            print(f"WARN: Could not parse action code: '{action_code}'")
            return None, None

    def step(self, action_code: str):
        """Takes action code, parses, executes, steps sim, gets obs, checks success."""
        print(f"\n--- Env Step: Received Action Code ---\n{action_code}\n---------------------------------")

        function_name, args = self._parse_action_code(action_code)
        action_success = False
        action_message = "Failed to parse action."
        info = {'action_code': action_code, 'parsed_function': None, 'parsed_args': None} # Initialize info

        if function_name:
            info['parsed_function'] = function_name
            info['parsed_args'] = args
            action_success, action_message = self.interface.execute_action(function_name, args)
        # else: Parsing failed message already set

        # Step simulation regardless of action outcome to advance time/state
        raw_obs_dict = self.interface.step_simulation()
        info['action_success'] = action_success
        info['action_message'] = action_message

        if raw_obs_dict is None: # Handle simulation step error
            print("ERROR: Simulation step failed. Cannot get observation or check success.")
            # Return current state? Or indicate error? Let's return empty state and mark done=True
            return "Error during simulation step", 0.0, True, info # reward 0, done=True to stop loop

        # Get new observation and check goal
        observation = self.interface.get_observation(raw_obs_dict)
        done = self.interface.check_success()
        reward = 1.0 if done else 0.0

        return observation, reward, done, info

    def close(self):
        self.interface.close()

    def get_task_goal_description(self):
        return self.interface.get_task_goal_description()
