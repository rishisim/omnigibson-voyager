# omnigibson-voyager
your_project_root/
|-- config.py                 # Configuration settings (paths, model names)
|-- omnigibson_interface.py   # Handles direct interaction with OmniGibson API
|-- omnigibson_env.py         # Voyager-style environment wrapper using the interface
|-- llm_api.py                # Wrapper for loading and querying DeepSeek
|-- run_voyager_omnigibson.py # Main script to run the agent loop
|-- prompts/                  # Directory for prompt templates
|   |-- basic_action_prompt.txt
|-- tasks/                    # Directory for task definitions (simplified)
|   |-- turn_on_light.yaml    # Example task definition
