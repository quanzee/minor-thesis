from openai import AzureOpenAI, OpenAI
from dotenv import load_dotenv
import os
import time
import openai


load_dotenv()

MODEL_NAME = "gpt-4o-mini"
MODEL_NAME_3 = "DeepSeek-V4-Flash"
MODEL_NAME_4 = "grok-4-1-fast-non-reasoning"

client = AzureOpenAI(
    azure_endpoint = os.getenv("AZURE_ENDPOINT"),
    api_key = os.getenv("AZURE_API_KEY"),
    api_version = os.getenv("AZURE_API_VERSION"),
    timeout=60.0,
    max_retries=0
)

deepseek_client = OpenAI(
    base_url = os.getenv("DEEPSEEK_ENDPOINT"),
    api_key = os.getenv("DEEPSEEK_API_KEY"),
    timeout=120.0,
    max_retries=0
)

grok_client = OpenAI(
    base_url=os.getenv("GROK_ENDPOINT"),
    api_key=os.getenv("GROK_API_KEY"),
    timeout=120.0,
    max_retries=0
)


def api_call_with_retry(call_fn, max_retries=5, base_delay=5):
    for attempt in range(max_retries):
        try:
            return call_fn()
        except openai.BadRequestError as e:
            # content filter responses are not retried
            if hasattr(e, 'body') and isinstance(e.body, dict):
                if e.body.get('code') == 'content_filter':
                    print(f"      Content filter triggered, skipping call.")
                    return None
                # some providers report the filter in choices[0]
                choices = e.body.get('choices')
                if choices and isinstance(choices, list):
                    first_choice = choices[0]
                    if first_choice.get('finish_reason') == 'content_filter':
                        print(f"      Content filter triggered, skipping call.")
                        return None
            raise
        except (openai.APITimeoutError, openai.APIConnectionError):
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"      API timeout, retrying in {delay}s... ({attempt + 1}/{max_retries})")
            time.sleep(delay)
        except openai.RateLimitError:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"      Rate limit, retrying in {delay}s... ({attempt + 1}/{max_retries})")
            time.sleep(delay)
        except (openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError):
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"      API error, retrying in {delay}s... ({attempt + 1}/{max_retries})")
            time.sleep(delay)
