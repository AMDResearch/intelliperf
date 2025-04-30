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

APPS=reduce reduce_inst
ISA=reduce.isa reduce_inst.isa

INST_BASE=/opt/logduration
INCLUDE=-I ${INST_BASE}/include/kerneldb -I ${INST_BASE}/include/dh_comms -I /opt/rocm/include/hsa
NO_WARN=-Wno-unused-result


INST_LIB=${INST_BASE}/lib/libAMDGCNSubmitAddressMessages-rocm.so

%: %.hip
	hipcc ${NO_WARN} ${INCLUDE} -O3 -g -o $@ $<

%_inst: %.hip
	hipcc ${NO_WARN} ${INCLUDE} -fgpu-rdc -fpass-plugin=${INST_LIB} -O3 -g -o $@ $<

%.isa: %
	/opt/rocm/bin/roc-obj-ls -v $< 2> /dev/null | \
	grep "gfx906" | awk '{print $$3}' | \
	/opt/rocm/bin/roc-obj-extract -o - | \
	/opt/rocm/llvm/bin/llvm-objdump --line-numbers - > $@

apps: ${APPS}
isa: ${ISA}

clean:
	rm -f ${APPS} ${ISA}
