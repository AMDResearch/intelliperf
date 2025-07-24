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


import dspy
import requests
import logging


class LLM:
	def __init__(
		self,
		api_key: str,
		system_prompt: str,
		model: str = "dvue-aoai-001-o4-mini",
		provider: str = "https://llm-api.amd.com/azure",
	):
		self.api_key = api_key
		self.system_prompt = system_prompt
		self.model = model
		self.provider = provider.rstrip("/")

		# Determine provider
		if "amd.com" in self.provider:
			self.use_amd = True
			self.header = {"Ocp-Apim-Subscription-Key": api_key}
		else:
			self.use_amd = False
			self.lm = dspy.LM(f"{self.provider}/{self.model}", api_key=api_key, max_tokens=16000)
			dspy.configure(lm=self.lm)

	def ask(self, user_prompt: str, signature=None, answer_type: str = "optimized_code"):
		if self.use_amd:
			# AMD/Azure REST call
			body = {
				"messages": [
					{"role": "system", "content": self.system_prompt},
					{"role": "user", "content": user_prompt},
				],
				"max_Tokens": 100000,
				"max_Completion_Tokens": 100000,
			}
			url = f"{self.provider}/engines/{self.model}/chat/completions"
			resp = requests.post(url, json=body, headers=self.header)
			resp.raise_for_status()
			# This path does not support structured output or reasoning.
			# It returns the content and None for reasoning to maintain a consistent return type.
			return resp.json()["choices"][0]["message"]["content"], None

		dspy.context(description=self.system_prompt)
		
		is_simple_signature = False
		if signature is None:
			is_simple_signature = True
			signature = f"prompt: str -> {answer_type}: str"
		
		chain = dspy.ChainOfThought(signature)
		response = chain(prompt=user_prompt)
		
		reasoning = getattr(response, "reasoning", None)

		if is_simple_signature:
			answer = getattr(response, answer_type)
			return answer, reasoning

		# For complex signatures, return the whole Prediction object
		return response, reasoning 