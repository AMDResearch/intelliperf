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

	def ask(self, user_prompt: str, record_meta: str = None) -> str:
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
				},
			)

		# Initialize reasoning variable
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
			else:
				# DSPy path: use ChainOfThought with clear signature
				# Define signature mapping input prompt to optimized code
				dspy.context(description=self.system_prompt)
				signature = "prompt: str -> optimized_code: str"
				chain = dspy.ChainOfThought(signature)
				ct_response = chain(prompt=user_prompt)

				# Extract both the reasoning and the final answer
				response_content = getattr(ct_response, "optimized_code", str(ct_response))

				# Try to capture the reasoning/chain-of-thought steps
				reasoning = getattr(ct_response, "reasoning", None)

			# Log successful response with reasoning if available
			if self.logger:
				success_data = {
					"response": response_content,
					"response_length": len(response_content),
					"record_meta": record_meta,
				}
				if reasoning:
					success_data["reasoning"] = reasoning
					success_data["reasoning_type"] = "chain_of_thought"

				self.logger.record("llm_call_success", success_data)

			return response_content

		except Exception as e:
			# Log error
			if self.logger:
				self.logger.record(
					"llm_call_error", {"error": str(e), "error_type": type(e).__name__, "record_meta": record_meta}
				)
			# Re-raise the exception to maintain existing behavior
			raise
