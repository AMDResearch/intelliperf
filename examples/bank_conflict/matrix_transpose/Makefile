APPS=matrix_transpose matrix_transpose_inst
ISA=matrix_transpose.isa matrix_transpose_inst.isa

INST_BASE=/opt/logduration
INCLUDE=-I ${INST_BASE}/include/kerneldb -I ${INST_BASE}/include/dh_comms -I /opt/rocm/include/hsa
NO_WARN=-Wno-unused-result


INST_LIB=${INST_BASE}/lib/libAMDGCNMemTraceHip.so

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