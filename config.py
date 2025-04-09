# config.py
import os

# --- Simulation ---
# Path to the *MODIFIED* OctoGibson environment configuration YAML
# (Ensure task, reset_joint_pos, kv are commented out for now)
ENV_CONFIG_PATH = "octogibson/config/Octogibson.yaml"
DEFAULT_SCENE_ID = "Merom_1_int" # Defined in the YAML

# --- Task ---
TASK_CONFIG_DIR = "tasks"
DEFAULT_TASK = "turn_on_light" # Corresponds to turn_on_light.yaml

# --- LLM ---
# Adjust if your paths changed
TOKENIZER_PATH = "/scratch/rnsimhad/deepseek_tokenizer"
MODEL_PATH = "/hf_cache/models--deepseek-ai--DeepSeek-R1-Distill-Llama-70B/snapshots/b1c0b44b4369b597ad119a196caf79a9c40e141e" # <--- UPDATE HASH
QUANTIZATION_BITS = 4 # Or 8 or None

# --- Prompts ---
PROMPT_DIR = "prompts"
ACTION_PROMPT_TEMPLATE_NAME = "basic_action_prompt.txt"

# --- Agent ---
MAX_STEPS_PER_TASK = 20 # Reduced steps for initial testing
LLM_MAX_NEW_TOKENS = 50

# --- YAML Workarounds ---
# These settings are problematic in the default OctoGibson YAML for this OmniGibson version
# and should be commented out in the loaded ENV_CONFIG_PATH file.
YAML_COMMENT_OUT = ["task:", "reset_joint_pos:", "kv:"]

# Verification
if not os.path.exists(ENV_CONFIG_PATH):
    raise FileNotFoundError(f"Environment config file not found at: {ENV_CONFIG_PATH}")
if not os.path.exists(TOKENIZER_PATH):
     raise FileNotFoundError(f"Tokenizer not found at: {TOKENIZER_PATH}")
if not os.path.exists(os.path.dirname(MODEL_PATH)): # Check model directory
     raise FileNotFoundError(f"Model directory not found at: {os.path.dirname(MODEL_PATH)}")
if not os.path.exists(PROMPT_DIR):
     os.makedirs(PROMPT_DIR)
if not os.path.exists(TASK_CONFIG_DIR):
     os.makedirs(TASK_CONFIG_DIR)

print("DEBUG [config.py]: Config loaded. Ensure problematic keys are commented in YAML.")
