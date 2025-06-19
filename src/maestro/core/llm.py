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


import requests

# first try your original import…
try:
	from dspy import DSPClient, PromptOptimizer
except (ImportError, ModuleNotFoundError):
	# …then fall back to `dsp`
	try:
		from dsp import DSPClient, PromptOptimizer
	except (ImportError, ModuleNotFoundError):
		DSPClient = None
		PromptOptimizer = None


class LLM:
	def __init__(
		self,
		api_key: str,
		system_prompt: str,
		deployment_id: str = "dvue-aoai-001-o4-mini",
		server: str = "https://llm-api.amd.com/azure",
	):
		self.api_key = api_key
		self.system_prompt = system_prompt
		self.deployment_id = deployment_id
		self.server = server.rstrip("/")
		self.header = {"Ocp-Apim-Subscription-Key": api_key}

		# route AMD traffic unchanged
		self.use_amd = "amd.com" in self.server

		if not self.use_amd:
			if DSPClient is None or PromptOptimizer is None:
				raise ImportError(
					"Could not import DSPClient/PromptOptimizer. "
					"Make sure you have installed the DSPy SDK (or `dsp`) and that "
					"these classes are exposed."
				)
			model_id = f"{self.server}/{self.deployment_id}"
			self.dsp_client = DSPClient(api_key=api_key, model=model_id)
			self.sys_optimizer = PromptOptimizer(role="system")
			self.user_optimizer = PromptOptimizer(role="user")

	def ask(self, user_prompt: str) -> str:
		if self.use_amd:
			body = {
				"messages": [
					{"role": "system", "content": self.system_prompt},
					{"role": "user", "content": user_prompt},
				],
				"max_Tokens": 4096,
				"max_Completion_Tokens": 4096,
			}
			resp = requests.post(
				url=f"{self.server}/engines/{self.deployment_id}/chat/completions",
				json=body,
				headers=self.header,
			)
			resp.raise_for_status()
			return resp.json()["choices"][0]["message"]["content"]

		# DSPy path
		optimized_system = self.sys_optimizer.optimize(self.system_prompt)
		optimized_user = self.user_optimizer.optimize(user_prompt)

		dsp_resp = self.dsp_client.chat(
			messages=[
				{"role": "system", "content": optimized_system},
				{"role": "user", "content": optimized_user},
			],
			max_tokens=4096,
		)
		return dsp_resp.choices[0].message.content
