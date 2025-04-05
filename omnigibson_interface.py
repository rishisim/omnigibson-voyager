# omnigibson_interface.py
import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import choose_from_options
from omnigibson.object_states import ToggledOn # Import necessary states
import yaml
import os
import time # For potential delays if needed

import config # Import our configuration

# Disable texture randomization for consistency
# gm.FORCE_TEXTURE_RANDOMIZATION = False # Might be useful for reproducibility

class OmniGibsonInterface:
    def __init__(self, scene_id=config.DEFAULT_SCENE_ID):
        print("Initializing OmniGibson Interface...")
        self.cfg = {
            "scene": {
                "type": "InteractiveTraversableScene",
                "scene_model": scene_id,
                # Add other scene configurations if needed
            },
            "robots": [
                {
                    "type": "Fetch",
                    "obs_modalities": ["scan", "rgb", "depth"], # Base modalities
                    "action_type": "continuous",
                    "action_normalize": True,
                }
            ],
            # Add other environment configurations if needed (physics, rendering)
        }
        self.env = None # Initialize later in load_task
        self.task_config = None
        self.robot = None
        self._initialize_env()
        print("OmniGibson Interface Initialized.")

    def _initialize_env(self):
        """Creates the OmniGibson environment instance."""
        print("Creating OmniGibson Environment (this might take a moment)...")
        try:
            # Setting dataset path globally if necessary (check OmniGibson docs)
            # og.set_og_dataset_path(config.OMNIGIBSON_DATASET_PATH) # May not be needed depending on install
            self.env = og.Environment(configs=self.cfg)
            # Warm-up steps might be needed for physics stabilization
            # for _ in range(20):
            #    self.env.step(action=np.zeros(self.env.action_dim))
            print("OmniGibson Environment created.")
        except Exception as e:
            print(f"ERROR: Failed to initialize OmniGibson environment: {e}")
            print("Check OmniGibson installation, dataset paths, and scene availability.")
            raise

    def load_task(self, task_name=config.DEFAULT_TASK):
        """Loads task config and resets the environment."""
        print(f"Loading task: {task_name}")
        task_path = os.path.join(config.TASK_CONFIG_DIR, f"{task_name}.yaml")
        if not os.path.exists(task_path):
            raise FileNotFoundError(f"Task configuration not found: {task_path}")

        with open(task_path, 'r') as f:
            self.task_config = yaml.safe_load(f)

        # You might need to update self.cfg based on task_config['scene_id']
        # and re-initialize the environment if the scene changes.
        # For simplicity now, we assume the scene matches the initial config.
        if self.task_config['scene_id'] != self.cfg['scene']['scene_model']:
             print(f"Warning: Task scene '{self.task_config['scene_id']}' differs from initial config '{self.cfg['scene']['scene_model']}'. Scene may not be correct.")
             # TODO: Implement logic to reload environment with the correct scene if needed.
             # self.cfg['scene']['scene_model'] = self.task_config['scene_id']
             # self._initialize_env() # This would restart the sim

        print("Resetting environment for task...")
        # Reset returns the first observation
        obs_dict, _ = self.env.reset()
        self.robot = self.env.robots[0] # Get robot instance
        print("Environment reset complete.")
        return self.get_observation(obs_dict) # Return formatted initial observation

    def get_observation(self, obs_dict=None):
        """Gathers state information and formats it for the LLM."""
        if self.env is None: return "Environment not initialized."
        if self.robot is None: return "Robot not found in environment."

        # If obs_dict is not provided, get it from the environment (might be less efficient)
        # if obs_dict is None:
        #    obs_dict = self.env.get_obs()

        # --- Basic Observation Example ---
        # TODO: Enhance this significantly. Needs more context.
        robot_pos = self.robot.get_position()
        nearby_objects = []
        max_dist = 3.0 # Objects within 3 meters

        # Iterate through potentially relevant objects
        # This is very basic - needs refinement based on task goals!
        for obj_cat in ["ceilings", "walls", "floors", "light_switches", "tables", "chairs", "counters"]: # Example categories
             if obj_cat in self.env.scene.objects_by_category:
                 for obj in self.env.scene.objects_by_category[obj_cat]:
                    try:
                        obj_pos = obj.get_position()
                        dist = ((robot_pos[0] - obj_pos[0])**2 + (robot_pos[1] - obj_pos[1])**2)**0.5
                        if dist < max_dist:
                            state_str = ""
                            # Check for specific states if the object has them
                            if obj.states and ToggledOn in obj.states:
                                is_on = obj.states[ToggledOn].get_value()
                                state_str = f" (state={'on' if is_on else 'off'})"
                            # Add more state checks (Open, Cooked, etc.) as needed

                            nearby_objects.append(f"{obj.name}{state_str}")
                    except Exception as e:
                        # Some objects might not have positions or states readily available
                        # print(f"Debug: Could not get info for object {obj.name}: {e}")
                        pass


        obs_text = f"Robot is at position ({robot_pos[0]:.2f}, {robot_pos[1]:.2f}).\n"
        if nearby_objects:
             obs_text += "Nearby objects: " + ", ".join(nearby_objects) + ".\n"
        else:
             obs_text += "No relevant objects detected nearby.\n"
        # Add inventory if robot can hold things:
        # inventory_text = "Robot is holding: ..."
        # obs_text += inventory_text

        return obs_text.strip()

    def get_task_goal_description(self):
        """Returns the natural language description of the goal."""
        if not self.task_config:
            return "No task loaded."
        return self.task_config.get("description", "No description provided.")

    def execute_action(self, function_name: str, args: list):
        """Executes a specific action based on parsed LLM output."""
        print(f"Attempting to execute action: {function_name} with args: {args}")
        if self.env is None or self.robot is None:
            print("ERROR: Environment or robot not ready for action.")
            return False, "Environment not ready." # Failed, Reason

        success = False
        message = ""
        try:
            if function_name == "navigate_to_object":
                object_name = args[0]
                # --- Placeholder ---
                print(f"PLACEHOLDER: Would navigate to {object_name}")
                # TODO: Implement actual navigation using robot controller
                # Find object, use motion planning (e.g., robot.controllers['navigation'].navigate_to_obj(...))
                # Need to handle finding the object instance first:
                # target_obj = self.env.scene.object_registry("name", object_name)
                # if target_obj: ... else: message = f"Object {object_name} not found."
                time.sleep(1) # Simulate action time
                success = True # Assume success for placeholder
                message = f"Navigation placeholder for {object_name} complete."

            elif function_name == "toggle_object":
                object_name = args[0]
                # --- Placeholder ---
                print(f"PLACEHOLDER: Would toggle {object_name}")
                # TODO: Implement actual toggling
                # target_obj = self.env.scene.object_registry("name", object_name)
                # if target_obj and ToggledOn in target_obj.states:
                #     current_state = target_obj.states[ToggledOn].get_value()
                #     target_obj.states[ToggledOn].set_value(not current_state) # This might need specific controller action
                #     success = True
                #     message = f"Toggled {object_name}."
                # else: message = f"Object {object_name} not found or cannot be toggled."
                time.sleep(0.5)
                success = True # Assume success for placeholder
                message = f"Toggle placeholder for {object_name} complete."

            elif function_name == "pick_up_object":
                object_name = args[0]
                 # --- Placeholder ---
                print(f"PLACEHOLDER: Would pick up {object_name}")
                # TODO: Implement grasping
                # Requires navigation first, then using manipulation controller
                # E.g., robot.controllers['arm_0'].grasp(...)
                success = True # Assume success for placeholder
                message = f"Pickup placeholder for {object_name} complete."

            elif function_name == "place_object_on":
                object_to_place = args[0] # Assumes robot is holding this (need inventory check)
                surface_object_name = args[1]
                # --- Placeholder ---
                print(f"PLACEHOLDER: Would place {object_to_place} on {surface_object_name}")
                # TODO: Implement placing
                # Requires navigation, then manipulation controller
                # E.g., robot.controllers['arm_0'].place_on(...)
                success = True # Assume success for placeholder
                message = f"Place placeholder for {object_to_place} on {surface_object_name} complete."

            else:
                message = f"Unknown action function: {function_name}"
                print(f"ERROR: {message}")
                success = False

        except IndexError:
            message = f"Incorrect number of arguments for action {function_name}."
            print(f"ERROR: {message}")
            success = False
        except Exception as e:
            message = f"Error executing action {function_name}: {e}"
            print(f"ERROR: {message}")
            success = False

        return success, message # Return success status and a message

    def step_simulation(self):
        """Steps the simulation forward one step."""
        if self.env is None:
            print("WARN: Cannot step, environment not initialized.")
            return None # Return None or empty dict if env not ready

        # In this setup, actions are executed via execute_action,
        # so we might just pass zero action here.
        # Or, if actions were continuous/low-level, they'd be passed here.
        action = self.robot.action_space.sample() * 0 # Zero action for now
        try:
            obs_dict, reward, terminated, truncated, info = self.env.step(action)
            # Note: Default reward/terminated/truncated might not be useful for LLM goals.
            # We rely on check_success instead.
            return obs_dict # Return the raw observation dictionary
        except Exception as e:
            print(f"ERROR during environment step: {e}")
            return None

    def check_success(self):
        """Checks if the task goal conditions are met."""
        if not self.task_config or 'goal_conditions' not in self.task_config:
            print("WARN: No goal conditions defined in task config.")
            return False # Cannot succeed if no goal is defined

        if self.env is None:
            print("WARN: Cannot check success, environment not initialized.")
            return False

        all_conditions_met = True
        for condition in self.task_config['goal_conditions']:
            obj_name = condition.get('object_name')
            condition_type = condition.get('type')
            target_state = condition.get('state', True) # Default to True state (e.g., On)

            if not obj_name or not condition_type:
                print(f"WARN: Invalid goal condition format: {condition}")
                all_conditions_met = False
                break

            # Find the object instance in the scene
            target_obj = self.env.scene.object_registry("name", obj_name)
            if not target_obj:
                print(f"WARN: Goal object '{obj_name}' not found in scene.")
                all_conditions_met = False
                break

            # Check the specific state based on type
            condition_met = False
            try:
                if condition_type == "ToggledOn":
                    if ToggledOn in target_obj.states:
                        current_state = target_obj.states[ToggledOn].get_value()
                        # Compare current state to the desired state (True for On, False for Off)
                        if isinstance(target_state, str): # Handle "on" / "off" string if used
                            target_bool = target_state.lower() == "on"
                        else:
                            target_bool = bool(target_state) # Treat non-string as boolean intention
                        condition_met = (current_state == target_bool)
                    else:
                        print(f"WARN: Object '{obj_name}' does not have ToggledOn state.")
                # Add checks for other OmniGibson state types here (e.g., Open, Position, InReach)
                # elif condition_type == "Open": ...
                # elif condition_type == "NextTo": ... # Check distance between objects
                else:
                    print(f"WARN: Unsupported goal condition type: {condition_type}")

            except Exception as e:
                 print(f"ERROR checking state for object '{obj_name}', condition '{condition_type}': {e}")
                 condition_met = False # Error means condition not met

            if not condition_met:
                all_conditions_met = False
                break # No need to check further conditions if one failed

        if all_conditions_met:
            print("SUCCESS: All task goal conditions met!")
        return all_conditions_met

    def close(self):
        """Closes the OmniGibson environment."""
        if self.env:
            print("Closing OmniGibson Environment.")
            self.env.close()
            self.env = None


----------------------------------------------- #TODO: Action: Carefully review the TODO comments, especially within execute_action. This is where you'll need to integrate the actual robot control logic using OmniGibson's API, likely referencing functions found in OctoGibson's robot_utils.py or directly using the robot controller objects (e.g., self.robot.controllers['navigation'], self.robot.controllers['arm_0']). Also refine get_observation to provide much more useful context.-----------------------
