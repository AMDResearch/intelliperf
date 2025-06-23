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

import os
from pathlib import Path

from intelliperf.utils.process import exit_on_fail


def get_guided_tuning_path():
	if os.environ.get("GT_TUNING"):
		return Path(os.environ["GT_TUNING"]).resolve()
	return (Path(__file__).resolve().parent / "../../../external/guided-tuning").resolve()


def get_accordo_path():
	return (Path(__file__).resolve().parent / "../../accordo").resolve()


def get_rocprofiler_path():
	return (Path(__file__).resolve().parent / "../../../external/rocprofiler-compute/src").resolve()


def get_nexus_path():
	return (Path(__file__).resolve().parent / "../../../external/nexus").resolve()


def get_llm_api_key():
	llm_key = os.environ.get("LLM_GATEWAY_KEY")
	if not llm_key:
		exit_on_fail(
			success=False,
			message="Missing LLM API key. Please set the LLM_GATEWAY_KEY environment variable.",
		)
	return llm_key
