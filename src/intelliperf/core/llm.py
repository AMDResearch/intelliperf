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
			self.lm = dspy.LM(f"{self.provider}/{self.model}", api_key=api_key)
			dspy.configure(lm=self.lm)

	def ask(self, user_prompt: str, signature="prompt: str -> optimized_code: str", answer_type: str = "optimized_code", record_meta: str = None):
		# Log the LLM interaction start
		if self.logger:
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

			# DSPy path
			dspy.context(description=self.system_prompt)
			chain = dspy.ChainOfThought(signature)
			ct_response = chain(prompt=user_prompt)

			# Try to capture reasoning if available (not returned)
			reasoning = getattr(ct_response, "reasoning", None)

			# Determine what to return based on signature type
			if isinstance(signature, str):
				# Simple signature: extract the requested answer_type field
				response_content = getattr(ct_response, answer_type, str(ct_response))
			else:
				# Complex signature (e.g., dspy.Signature subclass): return full prediction object
				response_content = ct_response

			# Log successful response
			if self.logger:
				log_payload = {
					"record_meta": record_meta,
					"signature": str(signature),
					"answer_type": answer_type,
				}
				try:
					if isinstance(response_content, str):
						log_payload["response"] = response_content
						log_payload["response_length"] = len(response_content)
					else:
						# Best-effort: record fields present on prediction object
						log_payload["response_fields"] = list(getattr(ct_response, "__dict__", {}).keys())
				except Exception:
					pass
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
					"llm_call_error", {"error": error_message, "error_type": error_type, "record_meta": record_meta}
				)

			print(f"ERROR: {error_message}")
			sys.exit(1)
