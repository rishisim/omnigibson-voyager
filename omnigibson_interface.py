# omnigibson_interface.py
import omnigibson as og
from omnigibson.macros import gm
from omnigibson import object_states
import yaml
import os
import time
import numpy as np
import sys

import config # Import our configuration
# Import utilities (ensure action_utils.py has the typo fixed)
from octogibson.utils import action_utils as au

class OmniGibsonInterface:
    def __init__(self):
        print("Initializing OmniGibson Interface...")
        self._load_config()
        self.env = None
        self.robot = None
        self.task_config = None
        self.action_dim = 0 # Store action dim here
        self._initialize_env()
        print("OmniGibson Interface Initialized.")

    def _load_config(self):
        """Loads and potentially modifies the environment config."""
        print(f"Loading environment config from: {config.ENV_CONFIG_PATH}")
        try:
            self.cfg = yaml.load(open(config.ENV_CONFIG_PATH, "r"), Loader=yaml.FullLoader)
            print("Initial config loaded.")
            # NOTE: We rely on the user having commented out problematic keys in the YAML
            # based on config.YAML_COMMENT_OUT and previous debugging.
            # Add timesteps if missing (required by og.Environment)
            if 'action_timestep' not in self.cfg:
                 self.cfg['action_timestep'] = float(1.0 / 60.0)
            else:
                 self.cfg['action_timestep'] = float(self.cfg['action_timestep'])
            if 'physics_timestep' not in self.cfg:
                 self.cfg['physics_timestep'] = float(1.0 / 60.0)
            else:
                 self.cfg['physics_timestep'] = float(self.cfg['physics_timestep'])
            print("Timesteps added/verified in config.")
        except Exception as e:
            print(f"FATAL: Error loading or parsing config YAML: {e}")
            raise

    def _initialize_env(self):
        """Creates the OmniGibson environment instance."""
        print("Creating OmniGibson Environment (this might take a moment)...")
        try:
            print(f"\nDEBUG: Final config being passed to og.Environment:\n{self.cfg}\n")
            self.env = og.Environment(configs=self.cfg)
            print("OmniGibson Environment created.")
            # Get robot and action dim immediately if environment loaded
            if self.env.robots:
                self.robot = self.env.robots[0]
                print(f"Robot instance acquired during init: {self.robot.name}")
                if hasattr(self.robot, 'action_dim'): # Check if controllers loaded enough
                     self.action_dim = self.robot.action_dim
                     print(f"Action dimension set: {self.action_dim}")
                else:
                     # Controllers might not be fully loaded (e.g., if controller_config is still commented)
                     print("WARN: Robot loaded but action_dim not available. Controllers might be missing/incomplete in config.")
                     self.action_dim = 0 # Default to 0 if not available
            else:
                print("WARN: No robot found in the environment after init.")
                self.action_dim = 0

            # Basic stabilization step if possible
            if self.action_dim > 0:
                print("Performing initial stabilization steps...")
                for _ in range(5): # Reduced steps
                    self.env.step(np.zeros(self.action_dim))
                print("Stabilization complete.")
            else:
                 print("Skipping stabilization steps (action_dim not available).")

        except Exception as e:
            print(f"FATAL: Failed to initialize OmniGibson environment: {e}")
            if self.env: self.env.close()
            raise

    def load_task(self, task_name=config.DEFAULT_TASK):
        """Loads task config and resets the environment."""
        print(f"Loading task: {task_name}")
        task_path = os.path.join(config.TASK_CONFIG_DIR, f"{task_name}.yaml")
        if not os.path.exists(task_path):
            raise FileNotFoundError(f"Task configuration not found: {task_path}")

        with open(task_path, 'r') as f:
            self.task_config = yaml.safe_load(f)

        # Optional: Check if scene_id in task matches env config
        # For now, assume they match or env is already loaded with correct scene

        print("Resetting environment for task...")
        obs_dict, info = self.env.reset() # Use reset method
        # Re-acquire robot instance and action_dim after reset
        self.robot = self.env.robots[0]
        if hasattr(self.robot, 'action_dim'):
            self.action_dim = self.robot.action_dim
        else: self.action_dim = 0 # Fallback if controllers not fully loaded
        print(f"Environment reset complete. Robot: {self.robot.name}, Action Dim: {self.action_dim}")
        return self.get_observation(obs_dict)

    def get_observation(self, obs_dict=None):
        """Gathers state information and formats it for the LLM."""
        if self.env is None or self.robot is None: return "Environment/Robot not ready."

        # Get current robot pose
        try:
            robot_pos, robot_orn = self.robot.get_position_orientation()
        except Exception as e:
            print(f"WARN: Could not get robot pose: {e}")
            robot_pos = np.array([0,0,0]) # Default if error

        # --- Enhanced Observation ---
        obs_lines = []
        obs_lines.append(f"Robot is at position ({robot_pos[0]:.2f}, {robot_pos[1]:.2f}).")

        # Inventory - TODO: Implement proper inventory tracking if needed
        # if hasattr(self.robot, 'inventory') and self.robot.inventory:
        #    obs_lines.append(f"Robot is holding: {', '.join(self.robot.inventory)}")
        # else:
        #    obs_lines.append("Robot is holding: Nothing.")
        obs_lines.append("Robot is holding: Nothing.") # Placeholder

        # Nearby Objects and States
        nearby_objects_str = []
        max_dist = 4.0 # Look a bit further

        try:
            all_objects = self.env.scene.objects # Use the list we verified
            if not all_objects:
                 print("WARN: env.scene.objects is empty.")

            for obj in all_objects:
                if obj == self.robot or not hasattr(obj, 'name') or not hasattr(obj, 'get_position_orientation'):
                    continue # Skip robot itself or objects without expected attributes

                try:
                    obj_pos, _ = obj.get_position_orientation()
                    dist = np.linalg.norm(robot_pos - obj_pos)

                    if dist < max_dist:
                        state_strs = []
                        # Check relevant states
                        if object_states.ToggledOn in obj.states:
                            state_strs.append(f"toggled={'on' if obj.states[object_states.ToggledOn].get_value() else 'off'}")
                        if object_states.Open in obj.states:
                            state_strs.append(f"open={'true' if obj.states[object_states.Open].get_value() else 'false'}")
                        # Add more states as needed (Cooked, Frozen, etc.)

                        state_desc = f" ({', '.join(state_strs)})" if state_strs else ""
                        nearby_objects_str.append(f"{obj.name}{state_desc} [{dist:.1f}m]")

                except Exception as obj_e:
                    # print(f"Debug: Error processing object {getattr(obj, 'name', 'N/A')}: {obj_e}") # Optional debug
                    pass # Skip object if error occurs

            if nearby_objects_str:
                obs_lines.append("Nearby objects: " + ", ".join(sorted(nearby_objects_str)))
            else:
                obs_lines.append("No relevant objects detected nearby.")

        except Exception as e:
            print(f"ERROR getting observation details: {e}")
            obs_lines.append("Error retrieving object details.")

        return "\n".join(obs_lines)

    def get_task_goal_description(self):
        """Returns the natural language description of the goal."""
        return self.task_config.get("description", "No description provided.") if self.task_config else "No task loaded."

    def execute_action(self, function_name: str, args: list):
        """Executes a specific action based on parsed LLM output."""
        print(f"Attempting to execute action: {function_name} with args: {args}")
        if self.env is None or self.robot is None:
            return False, "Environment or robot not ready."

        success = False
        message = ""
        try:
            target_obj_name = args[0] if args else None
            target_obj = self.env.scene.object_registry("name", target_obj_name) if target_obj_name else None

            if target_obj_name and target_obj is None:
                 message = f"Object '{target_obj_name}' not found in the scene."
                 print(f"ERROR: {message}")
                 return False, message

            # --- Action Implementations (Placeholders & TODOs) ---
            if function_name == "navigate_to_object":
                if target_obj:
                    print(f"TODO: Implement navigation to {target_obj.name} using robot controllers.")
                    # E.g., self.robot.controllers["base"]... or self.robot.controllers["navigation"]...
                    # Simulate time/success for now
                    time.sleep(1.0)
                    success = True
                    message = f"Navigation placeholder successful for {target_obj.name}."
                else: message = "Navigation requires a valid object name."

            elif function_name == "toggle_object":
                 if target_obj:
                     print(f"Attempting to toggle {target_obj.name}...")
                     if object_states.ToggledOn in target_obj.states:
                         # Option 1: Use utility (Less realistic, but simpler to start)
                         current_val = target_obj.states[object_states.ToggledOn].get_value()
                         target_val = not current_val
                         au.change_states(target_obj, "toggleable", int(target_val))
                         success = True
                         message = f"Toggled {target_obj.name} using utility function (current: {target_val})."

                         # Option 2: Use Controller (More realistic, requires controller setup)
                         # print(f"TODO: Implement toggling {target_obj.name} using robot controllers (IK, gripper).")
                         # Requires moving arm to switch, using gripper. Complex.
                         # success = True # Assume success for placeholder
                         # message = f"Controller toggle placeholder for {target_obj.name}."
                     else:
                          message = f"Object {target_obj.name} does not have Toggledon state."
                 else: message = "Toggle requires a valid object name."

            elif function_name == "pick_up_object":
                 if target_obj:
                    print(f"TODO: Implement picking up {target_obj.name} using robot controllers (arm, gripper).")
                    # Check distance, navigate close if needed, use IK, grasp action.
                    time.sleep(0.5)
                    success = True # Placeholder
                    message = f"Pickup placeholder for {target_obj.name}."
                 else: message = "Pickup requires a valid object name."

            elif function_name == "place_object_on":
                 if len(args) < 2: message = "Place requires object_to_place and surface_object_name."
                 else:
                     # object_to_place = args[0] # Need to check inventory
                     surface_obj_name = args[1]
                     surface_obj = self.env.scene.object_registry("name", surface_obj_name)
                     if not surface_obj: message = f"Surface object '{surface_obj_name}' not found."
                     # elif object_to_place not in self.robot.inventory: message = f"Robot not holding '{object_to_place}'."
                     else:
                         print(f"TODO: Implement placing object on {surface_obj.name} using robot controllers.")
                         # Navigate, use IK, release action.
                         time.sleep(0.5)
                         success = True # Placeholder
                         message = f"Place placeholder on {surface_obj.name}."

            else:
                message = f"Unknown action function: {function_name}"

        except IndexError:
            message = f"Incorrect number of arguments for action {function_name}."
        except Exception as e:
            message = f"Error executing action {function_name}: {e}"
            print(f"ERROR: {message}") # Print detailed error
            success = False # Ensure failure on exception

        print(f"Action Result: Success={success}, Msg='{message}'")
        return success, message

    def step_simulation(self):
        """Steps the simulation forward one step with zero action."""
        if self.env is None: return None
        if self.action_dim <= 0:
            print("WARN: Cannot step simulation, action_dim is 0. No controllers likely loaded.")
            # Returning dummy values, might need better handling
            return {}, 0, False, False, {"error": "action_dim is 0"}

        action = np.zeros(self.action_dim)
        try:
            # Step the environment
            obs_dict, reward, terminated, truncated, info = self.env.step(action)
            return obs_dict # Return new observations
        except Exception as e:
            print(f"ERROR during environment step: {e}")
            return None # Indicate failure

    def check_success(self):
        """Checks if the task goal conditions are met."""
        if not self.task_config or 'goal_conditions' not in self.task_config:
            return False # Cannot succeed if no goal is defined
        if self.env is None: return False

        all_conditions_met = True
        for condition in self.task_config['goal_conditions']:
            obj_name = condition.get('object_name')
            condition_type_str = condition.get('type') # e.g., "ToggledOn"
            target_state_val = condition.get('target_value', True) # Default target is True

            if not obj_name or not condition_type_str: continue # Skip invalid condition

            target_obj = self.env.scene.object_registry("name", obj_name)
            if not target_obj:
                all_conditions_met = False; break # Object not found

            # Map string condition type to OmniGibson state class
            condition_state_class = getattr(object_states, condition_type_str, None)
            if condition_state_class is None or condition_state_class not in target_obj.states:
                print(f"WARN: Condition type '{condition_type_str}' not found or not applicable to '{obj_name}'.")
                all_conditions_met = False; break

            # Check the actual state value
            try:
                current_state_val = target_obj.states[condition_state_class].get_value()
                # Handle boolean comparison carefully (e.g., 1 == True, 0 == False)
                target_bool = bool(target_state_val)
                current_bool = bool(current_state_val)
                if current_bool != target_bool:
                    all_conditions_met = False; break
            except Exception as e:
                 print(f"ERROR checking state {condition_type_str} for {obj_name}: {e}")
                 all_conditions_met = False; break

        if all_conditions_met: print("SUCCESS: Task goal conditions met!")
        return all_conditions_met

    def close(self):
        """Closes the OmniGibson environment."""
        if self.env:
            print("Closing OmniGibson Environment...")
            self.env.close()
            self.env = None
