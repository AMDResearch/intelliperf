import subprocess
import tempfile
import os
import time
import openai
from openai import OpenAIError


# Compiles a code string and returns success and binary path pair
def compiler_agent(code: str) -> tuple[bool, str]:
    with tempfile.NamedTemporaryFile(suffix=".hip", delete=False) as temp_file:
        temp_file.write(code.encode("utf-8"))
        temp_file_path = temp_file.name

    with tempfile.NamedTemporaryFile(suffix=".out", delete=False) as output_file:
        output_file_path = output_file.name

    try:
        compile_cmd = ["hipcc", temp_file_path, "-o", output_file_path]
        result = subprocess.run(
            compile_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        success = result.returncode == 0
        return success, (
            output_file_path if success else (result.stdout + result.stderr).strip()
        )
    except subprocess.CalledProcessError as e:
        return False, e.stdout + e.stderr


# Executes two binaries and validate updated one using the reference
def correctness_agent(reference: str, updated: str) -> tuple[bool, str]:
    ref_result = subprocess.run(
        reference, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    upd_result = subprocess.run(
        updated, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    success = ref_result.returncode == upd_result.returncode
    message = (
        "The code produced identical results"
        if success
        else "The code produced different results"
    )
    return success, message


# Simple performance agent that times wall-clock time
def performance_agent(reference: str, updated: str) -> tuple[bool, str]:
    start_ref = time.time()
    ref_result = subprocess.run(
        reference, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    ref_time = time.time() - start_ref

    start_upd = time.time()
    upd_result = subprocess.run(
        updated, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    upd_time = time.time() - start_upd

    success = ref_result.returncode == upd_result.returncode
    performant = upd_time < ref_time
    speedup = ref_time / upd_time
    if not success:
        return False, "The code didn't produce the same output."

    message = f"The code is {speedup}x faster. Old code took {ref_time} seconds and the optimized code took {upd_time} seconds."
    return performant, message


def optimizer_agent(prompt, temperature=0.0, max_tokens=3000) -> tuple[bool, str]:
    model = "gpt-4o"
    openai_key = os.getenv("OPENAI_API_KEY")

    if not openai_key:
        return False, "Error: Missing OpenAI API key."

    try:
        client = openai.Client(api_key=openai_key)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a skilled GPU HIP programmer. Given a kernel, you will optimize it and provide a correct performant implementation. Do not modify the kernel signature. Do not include any markdown code blocks or text other than the code.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return True, completion.choices[0].message.content.strip()

    except openai.AuthenticationError:
        return False, "Error: Authentication failed. Check your API key."
    except openai.RateLimitError:
        return False, "Error: Rate limit exceeded. Try again later."
    except openai.APIConnectionError:
        return False, "Error: Failed to connect to OpenAI API."
    except openai.OpenAIError as e:
        return False, f"Error: OpenAI API error - {str(e)}"
    except Exception as e:
        return False, f"Error: An unexpected error occurred - {str(e)}"


class OptimizerAgent:
    def __init__(
        self,
        model="gpt-4o",
        system_prompt="You are a skilled GPU HIP programmer. Given a kernel, you will optimize it and provide a correct performant implementation. Do not modify the kernel signature. Do not include any markdown code blocks or text other than the code.",
        temperature=0.7,
    ):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = model
        self.temperature = temperature
        self.messages = [{"role": "system", "content": system_prompt}]

    def chat(self, user_input):
        self.messages.append({"role": "user", "content": user_input})

        response = openai.ChatCompletion.create(
            model=self.model, messages=self.messages, temperature=self.temperature
        )

        assistant_reply = response["choices"][0]["message"]["content"]
        self.messages.append({"role": "assistant", "content": assistant_reply})

        return assistant_reply

    def reset_conversation(self):
        """Clear the message history while retaining the system prompt."""
        self.messages = [self.messages[0]]
