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


import importlib
import inspect

import dspy
import requests


def _find_client():
	"""Locate the DSPy LLM client class under dspy."""
	if hasattr(dspy, "DSPClient"):
		return dspy.DSPClient
	if hasattr(dspy, "LM"):
		return dspy.LM
	try:
		m = importlib.import_module("dspy.clients.lm")
		if hasattr(m, "DSPClient"):
			return m.DSPClient
		if hasattr(m, "LM"):
			return m.LM
	except ImportError:
		pass
	return None


def _find_optimizer():
	"""Locate the prompt optimizer class under dspy."""
	if hasattr(dspy, "PromptOptimizer"):
		return dspy.PromptOptimizer
	try:
		m = importlib.import_module("dspy.teleprompt")
		if hasattr(m, "PromptOptimizer"):
			return m.PromptOptimizer
		if hasattr(m, "BootstrapFewShot"):
			return m.BootstrapFewShot
	except ImportError:
		pass
	return None


DSPClient = _find_client()
PromptOptimizer = _find_optimizer()


class _NoOpPromptOptimizer:
	def __init__(self, *args, **kwargs):
		pass

	def optimize(self, prompt: str) -> str:
		return prompt


if DSPClient is None:
	raise ImportError(
		"Could not find a DSPy LLM client class under `dspy`. "
		"Please `pip install dspy-ai` and verify that `dspy.DSPClient` or `dspy.clients.lm.LM` is available."
	)

# Ensure the optimizer has an optimize() method; otherwise fall back to no-op
if PromptOptimizer is None or not hasattr(PromptOptimizer, "optimize"):
	PromptOptimizer = _NoOpPromptOptimizer


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

		self.use_amd = "amd.com" in self.server

		if not self.use_amd:
			model_id = f"{self.server}/{self.deployment_id}"

			sig = inspect.signature(DSPClient.__init__)
			params = sig.parameters
			if "api_key" in params and "model" in params:
				self.dsp_client = DSPClient(api_key=api_key, model=model_id)
			else:
				try:
					self.dsp_client = DSPClient(model_id, api_key=api_key)
				except TypeError:
					self.dsp_client = DSPClient(model_id)

			opt_params = inspect.signature(PromptOptimizer.__init__).parameters
			if "role" in opt_params:
				self.sys_optimizer = PromptOptimizer(role="system")
				self.user_optimizer = PromptOptimizer(role="user")
			else:
				self.sys_optimizer = PromptOptimizer()
				self.user_optimizer = PromptOptimizer()

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
			url = f"{self.server}/engines/{self.deployment_id}/chat/completions"
			resp = requests.post(url, json=body, headers=self.header)
			resp.raise_for_status()
			return resp.json()["choices"][0]["message"]["content"]

		# DSPy path: optimize prompts then call
		optimized_system = self.sys_optimizer.optimize(self.system_prompt)
		optimized_user = self.user_optimizer.optimize(user_prompt)

		dsp_resp = self.dsp_client(
			messages=[
				{"role": "system", "content": optimized_system},
				{"role": "user", "content": optimized_user},
			],
			max_tokens=4096,
		)

		# Handle different possible return shapes:
		#  - list of strings
		#  - list of dicts
		#  - dict with "choices"
		if isinstance(dsp_resp, list):
			first = dsp_resp[0]
			if isinstance(first, str):
				return first
			if isinstance(first, dict) and "choices" in first:
				return first["choices"][0]["message"]["content"]

		if isinstance(dsp_resp, dict) and "choices" in dsp_resp:
			return dsp_resp["choices"][0]["message"]["content"]

		# Fallback to string conversion
		return str(dsp_resp)
