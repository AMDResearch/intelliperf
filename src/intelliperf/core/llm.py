################################################################################
# MIT License

# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
################################################################################


import sys
from typing import Optional

import dspy
import requests

from intelliperf.core.logger import Logger


class LLM:
    def _get_model_context_length(self) -> Optional[int]:
        """Query the model's max context length from the API"""
        import logging

        try:
            # Try to get model info from OpenRouter
            if "openrouter" in self.provider.lower():
                models_url = "https://openrouter.ai/api/v1/models"
                headers = {"Authorization": f"Bearer {self.api_key}"}
                resp = requests.get(models_url, headers=headers, timeout=5)
                if resp.status_code == 200:
                    models_data = resp.json().get("data", [])
                    for model_info in models_data:
                        if model_info.get("id") == self.model:
                            context_length = model_info.get("context_length")
                            if context_length:
                                logging.info(
                                    f"Model {self.model} max context: {context_length:,} tokens"
                                )
                                return context_length

            # Try to get from litellm/dspy metadata if available
            if hasattr(self.lm, "model_info"):
                context_length = getattr(self.lm.model_info, "max_tokens", None)
                if context_length:
                    logging.info(
                        f"Model {self.model} max context: {context_length:,} tokens"
                    )
                    return context_length

        except Exception as e:
            logging.debug(f"Could not query model context length: {e}")

        return None

    def __init__(
        self,
        api_key: str,
        system_prompt: str,
        model: str = "dvue-aoai-001-o4-mini",
        provider: str = "https://llm-api.amd.com/azure",
        logger: Optional[Logger] = None,
    ):
        self.api_key = api_key
        self.system_prompt = system_prompt
        self.model = model
        self.provider = provider.rstrip("/")
        self.logger = logger

        # Determine provider
        if "amd.com" in self.provider:
            self.use_amd = True
            self.header = {"Ocp-Apim-Subscription-Key": api_key}
        else:
            self.use_amd = False
            # Query model context and reserve ~20% for input, rest for output
            max_context = self._get_model_context_length()
            max_output_tokens = int(max_context * 0.8) if max_context else 4096
            # Set timeout to 10 minutes (600 seconds)
            timeout_mins = 1
            self.lm = dspy.LM(
                f"{self.provider}/{self.model}",
                api_key=api_key,
                max_tokens=max_output_tokens,
                timeout=timeout_mins * 60,
            )
            dspy.configure(lm=self.lm)

    def ask(
        self,
        user_prompt: str = "",
        signature="prompt: str -> optimized_code: str",
        answer_type: str = "optimized_code",
        record_meta: str = None,
        **input_kwargs,
    ):
        """
        Ask the LLM a question using DSPy signatures.

        Args:
                user_prompt: For simple string-based signatures, the prompt text
                signature: DSPy signature (string or Signature class)
                answer_type: Which field to extract from response (for string signatures)
                record_meta: Metadata for logging
                **input_kwargs: For complex signatures with multiple inputs (e.g., kernel_code, history, etc.)
        """
        # Log the LLM interaction start
        if self.logger:
            # For input_kwargs, log the actual values
            logged_inputs = {}
            if input_kwargs:
                for key, value in input_kwargs.items():
                    # For history object, extract the messages list
                    if key == "history" and hasattr(value, "messages"):
                        logged_inputs[key] = value.messages
                    else:
                        logged_inputs[key] = value

            self.logger.record(
                "llm_call_start",
                {
                    "system_prompt": self.system_prompt,
                    "user_prompt": user_prompt,
                    "model": self.model,
                    "provider": self.provider,
                    "record_meta": record_meta,
                    "signature": str(signature),
                    "answer_type": answer_type,
                    "input_kwargs": logged_inputs if input_kwargs else None,
                },
            )

        reasoning = None

        try:
            if self.use_amd:
                # AMD/Azure REST call
                body = {
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "max_Tokens": 4096,
                    "max_Completion_Tokens": 4096,
                }
                url = f"{self.provider}/engines/{self.model}/chat/completions"
                resp = requests.post(url, json=body, headers=self.header)
                resp.raise_for_status()
                response_content = resp.json()["choices"][0]["message"]["content"]

                # Log successful response
                if self.logger:
                    self.logger.record(
                        "llm_call_success",
                        {
                            "response": response_content,
                            "response_length": len(response_content),
                            "record_meta": record_meta,
                        },
                    )

                return response_content
            else:  # DSPy path
                dspy.context(description=self.system_prompt)
                chain = dspy.ChainOfThought(signature)

                # Determine how to call the chain
                if input_kwargs:
                    # Complex signature with multiple inputs
                    ct_response = chain(**input_kwargs)
                else:
                    # Simple signature with just prompt
                    ct_response = chain(prompt=user_prompt)

                # Try to capture reasoning if available (not returned)
                reasoning = getattr(ct_response, "reasoning", None)

                # Determine what to return based on signature type
                if isinstance(signature, str):
                    # Simple signature: extract the requested answer_type field
                    response_content = getattr(
                        ct_response, answer_type, str(ct_response)
                    )
                else:
                    # Complex signature (e.g., dspy.Signature subclass): return full prediction object
                    response_content = ct_response

                # Log successful response
                if self.logger:
                    log_payload = {
                        "record_meta": record_meta,
                        "signature": str(signature),
                        "answer_type": answer_type,
                        "response": response_content,
                    }
                    if reasoning:
                        log_payload["reasoning"] = reasoning
                        log_payload["reasoning_type"] = "chain_of_thought"
                    self.logger.record("llm_call_success", log_payload)

                return response_content

        except Exception as e:
            error_message = str(e)
            error_type = type(e).__name__
            if self.logger:
                self.logger.record(
                    "llm_call_error",
                    {
                        "error": error_message,
                        "error_type": error_type,
                        "record_meta": record_meta,
                    },
                )
            print(f"ERROR: {error_message}")
            sys.exit(1)
