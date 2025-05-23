# llm_api.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import time
import config
import os # Added for path check

class LLM_API:
    def __init__(self, model_path=config.MODEL_PATH, tokenizer_path=config.TOKENIZER_PATH, quantization_bits=config.QUANTIZATION_BITS):
        print("Initializing LLM API...")
        if not os.path.exists(tokenizer_path): raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
        if not os.path.exists(model_path): raise FileNotFoundError(f"Model path not found: {model_path}") # Check specific model path

        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        if self.device == "cpu": print("Warning: Running LLM on CPU will be very slow!")

        self.bnb_config = None
        if quantization_bits == 4:
            print("Setting up 4-bit quantization...")
            self.bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        elif quantization_bits == 8:
            print("Setting up 8-bit quantization...")
            self.bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        else: print("No quantization configured.")

        self._load_model()
        print("LLM API Initialized.")

    def _load_model(self):
        print(f"Loading tokenizer from: {self.tokenizer_path}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True)
            print("Tokenizer loaded.")
        except Exception as e: print(f"ERROR loading tokenizer: {e}"); raise

        print(f"Loading model from: {self.model_path}...")
        print(f"Quantization config: {self.bnb_config}")
        start_time = time.time()
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=self.bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            print(f"Model loaded in {time.time() - start_time:.2f} seconds.")
        except Exception as e: print(f"ERROR loading model: {e}"); raise

        if self.tokenizer.pad_token is None:
            print("Setting pad_token to eos_token.")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            if hasattr(self.model, 'config'): # Ensure model config exists
                 self.model.config.pad_token_id = self.model.config.eos_token_id

    def generate(self, prompt: str, max_new_tokens=config.LLM_MAX_NEW_TOKENS):
        """Generates text based on the prompt."""
        print(f"\n--- Sending Prompt to LLM (length={len(prompt)}) ---")
        # print(prompt) # Uncomment for full prompt debugging
        print("-------------------------------------------------")

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4000).to(self.device) # Added max_length
            if inputs.input_ids.nelement() == 0: # Check if inputs are empty
                 print("WARN: Input tokens are empty, cannot generate.")
                 return ""

            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                 "pad_token_id": self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id, # Ensure pad_token_id is not None
                 "eos_token_id": self.tokenizer.eos_token_id,
                 # "temperature": 0.7, # Optional sampling params
                 # "do_sample": True,
            }

            print("Generating response...")
            start_time = time.time()
            with torch.no_grad():
                 output_ids = self.model.generate(**inputs, **gen_kwargs)
            generation_time = time.time() - start_time
            print(f"Response generated in {generation_time:.2f} seconds.")

            # Decode only the newly generated tokens
            output_text = self.tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

            print(f"\n--- LLM Raw Output ---\n{output_text}\n----------------------")
            return output_text.strip()

        except Exception as e:
            print(f"ERROR during LLM generation: {e}")
            return f"Error: {e}"
