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

from formulas.formula_base import Formula_Base

class atomic_contention(Formula_Base):
    def __init__(self, name: str, build_command: list, instrument_command: list, app_cmd: list, top_n: int):
        super().__init__(name, build_command, instrument_command, app_cmd, top_n)

    def profile_pass(self):
        return super().profile_pass()

    def instrument_pass(self):
        return super().instrument_pass()

    def optimize_pass(self):
        return super().optimize_pass()

    def compile_pass(self):
        return super().compile_pass()

    def validation_pass(self):
        return super().validation_pass()

    def source_code_pass(self):
        return super().source_code_pass()
