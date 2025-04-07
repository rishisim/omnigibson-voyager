# config.py
import os

# --- Simulation ---
# Example scene, find relevant ones in OmniGibson dataset documentation
DEFAULT_SCENE_ID = "Rs_int"
# Path to your OmniGibson dataset (replace with actual path if not default)
OMNIGIBSON_DATASET_PATH = os.environ.get("OMNIGIBSON_DATASET_PATH", "/data") # Assumes /data is mounted

# --- Task ---
TASK_CONFIG_DIR = "tasks"
DEFAULT_TASK = "turn_on_light" # Corresponds to turn_on_light.yaml

# --- LLM ---
# Adjust these paths to where you have the model and tokenizer stored
TOKENIZER_PATH = "/scratch/rnsimhad/deepseek_tokenizer" # Or your host path if binding directly
# Find the correct snapshot hash for your downloaded model
MODEL_PATH = "/scratch/rnsimhad/.hf_cache/models--deepseek-ai--DeepSeek-R1-Distill-Llama-70B/snapshots/b1c0b44b4369b597ad119a196caf79a9c40e141e"
# Make sure the snapshot hash above is correct!

# Use None if you don't want quantization (requires significantly more VRAM)
QUANTIZATION_BITS = 4 # Or 8 or None

# --- Prompts ---
PROMPT_DIR = "prompts"
ACTION_PROMPT_TEMPLATE_NAME = "basic_action_prompt.txt"

# --- Agent ---
MAX_STEPS_PER_TASK = 50

-----------------------------------#TODO (FINISHED): Action: Replace YOUR_MODEL_SNAPSHOT_HASH with the actual hash from your .hf_cache directory. Verify OMNIGIBSON_DATASET_PATH is correct for your setup.-------------------------
