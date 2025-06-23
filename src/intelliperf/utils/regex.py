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
import re


def generate_ecma_regex_from_list(kernel_names: set) -> str:
	res = []
	updated_kernel_names = set()
	for i in kernel_names:
		if " [clone .kd]" not in i:
			i += " [clone .kd]"
			updated_kernel_names.add(i)
	for i in updated_kernel_names:
		escaped_string = re.escape(i)
		regex_string = r"^" + escaped_string + r"$"
		res.append(regex_string)
		# Note: Temporary fix, but until bug in omniprobe is fixed we need to also
		# add the name of the instrumented kernel clone to the regex, otherwise we'll skip it
		# and exclude it from the memory analysis report
		# duplicate_kernel_str = f"__amd_crk_{i.replace(')', ', void*)', 1)}"
		# duplicate_kernel_str = f"__amd_crk_{i.replace(")", ", void*)", 1)}"
		# escaped_string = re.escape(duplicate_kernel_str)
		# regex_string = r"^" + escaped_string + r"$"
		# res.append(regex_string)

	regex = f"({'|'.join(res)})"
	return regex
