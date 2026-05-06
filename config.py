from openai import AzureOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

MODEL_NAME = "gpt-4o-mini"
client = AzureOpenAI(
    azure_endpoint = os.getenv("AZURE_ENDPOINT"), 
    api_key = os.getenv("AZURE_API_KEY"),  
    api_version = os.getenv("AZURE_API_VERSION")
)
