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

class LLM:
    def __init__(
        self,
        model: str,
        api_key: str,
        system_prompt: str,
        deployment_id: str = "dvue-aoai-001-o4-mini",
        server: str = "https://llm-api.amd.com/azure",
    ):
        self.model = model
        self.api_key = api_key
        self.system_prompt = system_prompt
        self.deployment_id = deployment_id
        self.server = server
        self.header = {"Ocp-Apim-Subscription-Key": api_key}
    def ask(self, user_prompt: str) -> str:
        body = {
            "messages": [
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {"role": "user", "content": user_prompt},
            ],
            "max_Tokens": 4096,
            "max_Completion_Tokens": 4096,
        }

        response = requests.post(
            url=f"{self.server}/engines/{self.deployment_id}/chat/completions",
            json=body,
            headers=self.header,
        ).json()
        return response['choices'][0]['message']['content']
