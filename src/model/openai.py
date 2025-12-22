import os
import itertools
import time
import openai
from openai import OpenAI, RateLimitError, APIError, APIConnectionError, APITimeoutError

from datetime import timedelta
import random

from typing import List, Dict, Union, Any, Generator
from tqdm import tqdm
from transformers import GPT2TokenizerFast

from src.model.model import Model
from src import cache


class OpenAIModel(Model):
    """
    Wrapper for calling OpenAI with some safety and retry loops as well as a somewhat advanced caching mechanism.
    """

    engine: str
    api_max_attempts: int
    api_endpoint: str

    max_tokens: int
    stop_token: str

    log_probs: int
    num_samples: int

    echo: bool

    temperature: float
    client: OpenAI

    def __init__(
            self,
            engine: str = 'text-davinci-003',
            api_max_attempts: int = 60,
            api_endpoint: str = 'completion',

            temperature: float = 1.0,
            top_p: float = 1.0,
            max_tokens: int = 2049,
            stop_token: str = None,
            log_probs: int = 1,
            num_samples: int = 1,
            echo: bool = True,

            prompt_cost: float = None,
            completion_cost: float = None

    ):
        """

        :param engine: The model you are calling
        :param api_max_attempts: Retry the api call N times.
        :param api_endpoint: Usually differs between completion or chat endpoint
        :param temperature: https://platform.openai.com/docs/api-reference/chat/create#chat/create-temperature
        :param top_p: https://platform.openai.com/docs/api-reference/chat/create#chat/create-top_p
        :param max_tokens: https://platform.openai.com/docs/api-reference/chat/create#chat/create-max_tokens
        :param stop_token: https://platform.openai.com/docs/api-reference/chat/create#chat/create-stop
        :param log_probs: (only for completion) https://platform.openai.com/docs/api-reference/completions/create#completions/create-logprobs
        :param num_samples: https://platform.openai.com/docs/api-reference/chat/create#chat/create-n
        :param echo: (only for completion) https://platform.openai.com/docs/api-reference/completions/create#completions/create-echo
        :param prompt_cost: Pass in the current cost of the api you are calling to track costs (optional)
        :param completion_cost: Pass in the current cost of the api you are calling to track costs (optional)
        """

        self.engine = engine

        self.api_max_attempts = api_max_attempts
        self.api_endpoint = api_endpoint.lower()

        self.max_tokens = max_tokens
        self.stop_token = stop_token

        self.log_probs = log_probs
        self.num_samples = num_samples

        self.echo = echo

        self.temperature = temperature
        self.top_p = top_p

        self.gpt_waittime = 60

        self.prompt_cost = prompt_cost
        self.completion_cost = completion_cost
        self.total_cost = 0.0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            # Try to get from openai.api_key if set elsewhere, though deprecated usage
            api_key = openai.api_key
            
        self.client = OpenAI(api_key=api_key)

    def __update_cost__(self, raw):
        usage = raw.get('usage', {})
        if usage:
            completion_tokens = usage.get('completion_tokens', 0)
            prompt_tokens = usage.get('prompt_tokens', 0)
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            if self.prompt_cost and self.completion_cost:
                cost = completion_tokens * self.completion_cost + prompt_tokens * self.prompt_cost
                self.total_cost += cost

    @cache.cached(data_ex=timedelta(days=30), no_data_ex=timedelta(hours=1), prepended_key_attr='engine,num_samples,log_probs,echo,temperature=float(0),top_p=float(1.0),stop_token,max_tokens')
    def inference(self, prompt: str, *args, **kwargs) -> Any:
        if self.api_endpoint == 'completion':
            out = self.__safe_openai_completion_call__(
                prompt,
                *args,
                **kwargs
            )
        elif self.api_endpoint == 'chat':
            out = self.__safe_openai_chat_call__(
                prompt,
                *args,
                **kwargs
            )
        else:
            raise Exception(f"Unknown api endpoint for openai model: {self.api_endpoint}")

        # For v1 objects, we can still access usage, but let's be safe
        # The cache decorator pickles the response, so if it's a pydantic model it should be fine.
        # But we might need to ensure we are returning a dict or similar if the rest of the code expects it.
        # The original code returned the raw response object from openai (except in error cases).
        # V1 returns pydantic models.
        
        # Update cost
        if not isinstance(out, dict) or not out.get("API Error"):
             self.__update_cost__(out)
             
        return out

    def __safe_openai_completion_call__(
            self,
            prompt: str,
            temperature: float = None,
            max_tokens: int = None,
            stop_token: str = None,
            logprobs: int = None,
            num_samples: int = None,
            echo: bool = None
    ) -> Any:
        if max_tokens is None:
            max_tokens = self.max_tokens
        if temperature is None:
            temperature = self.temperature
        if stop_token is None:
            stop_token = self.stop_token
        if logprobs is None:
            logprobs = self.log_probs
        if num_samples is None:
            num_samples = self.num_samples
        if echo is None:
            echo = self.echo

        last_exc = None
        for i in range(self.api_max_attempts):
            try:
                return self.client.completions.create(
                    model=self.engine,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    logprobs=logprobs,
                    n=num_samples,
                    echo=echo,
                    stop=stop_token
                ).model_dump()
            except RateLimitError as e:
                last_exc = e
                print(f"ERROR: OPENAI Rate Error: {e}")
                time.sleep(self.gpt_waittime + int(random.randint(1, 10)))
            except APIError as e:
                last_exc = e
                print(f"ERROR: OPENAI API Error: {e}")
            except APITimeoutError as e:
                last_exc = e
                print(f"ERROR: OPENAI Timeout Error: {e}")
            except APIConnectionError as e:
                last_exc = e
                print(f"ERROR: OPENAI APIConnection Error: {e}")
            except Exception as e:
                # Catch all other openai errors or general errors
                last_exc = e
                print(f"ERROR: OPENAI General Error: {e}")
                
        # make a fake response that matches expected structure
        return {
                "choices": [{"message": {"content": f"OPENAI Error - {str(last_exc)}"}}],
                "text": prompt + " OPENAI Error - " + str(last_exc),
                "API Error": True,
        }

    def __safe_openai_chat_call__(
            self,
            prompt: str,
            system_prompt: str = None,
            temperature: float = None,
            top_p: float = None,
            max_tokens: int = None,
            stop_token: str = None,
            num_samples: int = None,
    ) -> Any:
        if max_tokens is None:
            max_tokens = self.max_tokens
        if temperature is None:
            temperature = self.temperature
        if top_p is None:
            top_p = self.top_p
        if stop_token is None:
            stop_token = self.stop_token
        if num_samples is None:
            num_samples = self.num_samples

        last_exc = None
        for i in range(self.api_max_attempts):
            try:
                # TODO - look at different roles?
                messages = [
                    {"role": "user", "content": prompt}
                ]

                if system_prompt:
                    messages = [{'role': 'system', 'content': system_prompt}, {"role": "user", "content": prompt}]

                # Use max_completion_tokens for newer models (gpt-4o, o1, etc.)
                # Fall back to max_tokens for older models
                return self.client.chat.completions.create(
                    model=self.engine,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_completion_tokens=max_tokens,
                    n=num_samples,
                    stop=stop_token
                ).model_dump()
            except RateLimitError as e:
                last_exc = e
                print(f"ERROR: OPENAI Rate Error: {e}")
                time.sleep(self.gpt_waittime)
            except APIError as e:
                last_exc = e
                print(f"ERROR: OPENAI API Error: {e}")
            except APITimeoutError as e:
                last_exc = e
                print(f"ERROR: OPENAI Timeout Error: {e}")
            except APIConnectionError as e:
                last_exc = e
                print(f"ERROR: OPENAI APIConnection Error: {e}")
            except Exception as e:
                last_exc = e
                print(f"ERROR: OPENAI General Error: {e}")

        # make a fake response that matches expected structure
        return {
            "choices": [{"message": {"content": f"OPENAI Error - {str(last_exc)}"}}],
            "text": prompt + " OPENAI Error - " + str(last_exc),
            "API Error": True,
        }
