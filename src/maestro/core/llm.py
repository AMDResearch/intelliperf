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


class LLM:
	def __init__(
		self,
		api_key: str,
		system_prompt: str,
		deployment_id: str = "gpt-4o-mini",
		server: str = "https://llm-api.amd.com/azure",
	):
		self.api_key = api_key
		self.system_prompt = system_prompt
		self.deployment_id = deployment_id
		self.server = server.rstrip("/")

		# Determine provider
		if "amd.com" in self.server:
			self.use_amd = True
			self.header = {"Ocp-Apim-Subscription-Key": api_key}
		else:
			self.use_amd = False
			# Configure DSPy for OpenAI
			self.lm = dspy.LM(f"openai/{deployment_id}", api_key=api_key)
			dspy.configure(lm=self.lm)

	def ask(self, user_prompt: str) -> str:
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
			url = f"{self.server}/engines/{self.deployment_id}/chat/completions"
			resp = requests.post(url, json=body, headers=self.header)
			resp.raise_for_status()
			return resp.json()["choices"][0]["message"]["content"]

			# DSPy path: use ChainOfThought with clear signature
		# Define signature mapping input prompt to optimized code
		dspy.context(description=self.system_prompt)
		signature = "prompt: str -> optimized_code: str"
		chain = dspy.ChainOfThought(signature)
		ct_response = chain(prompt=user_prompt)
		# Return optimized code from 'answer' field
		return getattr(ct_response, "answer", str(ct_response))
