from phoenix.client import Client
from src.config import PHOENIX_COLLECTOR_ENDPOINT


# Initialize a phoenix client with your phoenix endpoint
# By default it will read from your environment variables
def get_prompt(prompt_name: str):

    client = Client()

    # Pulling a prompt by name
    prompt = client.prompts.get(prompt_identifier=prompt_name)
    return prompt