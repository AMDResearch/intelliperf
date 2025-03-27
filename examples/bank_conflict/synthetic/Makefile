APPS=bnk_conflict bnk_conflict_inst
ISA=bnk_conflict.isa bnk_conflict_inst.isa

INST_BASE=/opt/logduration
INCLUDE=-I ${INST_BASE}/include/kerneldb -I ${INST_BASE}/include/dh_comms -I /opt/rocm/include/hsa
NO_WARN=-Wno-unused-result


INST_LIB=${INST_BASE}/lib/libAMDGCNSubmitAddressMessages-rocm.so

%: %.cpp
	hipcc ${NO_WARN} ${INCLUDE} -O3 -g -o $@ $<

%_inst: %.cpp
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
