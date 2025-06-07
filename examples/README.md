# Examples


## Bank Conflict Formula

```console
maestro --project_directory=./examples\
    --build_command=../scripts/build_examples.sh\
    --instrument_command="../scripts/build_examples.sh -i -c"\
    --formula=bankConflict -- ./build/bank_conflict/matrix_transpose
```

## Memory Access Forumla

```console
maestro --project_directory=./examples\
    --build_command=../scripts/build_examples.sh\
    --instrument_command="../scripts/build_examples.sh -i -c"\
    --formula=memoryAccess -- ./build/access_pattern/uncoalesced
```

## Atomic Contention Formula

```console
maestro --project_directory=./examples\
    --build_command=../scripts/build_examples.sh\
    --instrument_command="../scripts/build_examples.sh -i -c"\
    --formula=atomicContention -- ./build/contention/reduction
```


Example output:

## Memory Access Forumla


After running, you will get an output similar to:

```console
Transpose correct ✅
Transpose correct ✅
{
  "optimized": [
    {
      "name": "2025_06_04-04_57_29",
      "kernel": "matrix_transpose(float const*, float*, int, int)",
      "count": 1,
      "gpu_series": "MI300",
      "cu_per_gpu": 304,
      "max_sclk": 2100,
      "hbm_bw": 8601.6,
      "lds_banks_per_cu": 32,
      "total_l2_chan": 128,
      "se_per_gpu": 32,
      "num_xcd": 8,
      "grid": 1048576.0,
      "workgroup": 256.0,
      "durations": {
        "ns": 6736.0,
        "pct": 100.0
      },
      "flops": {
        "f": 0.0,
        "i": 17825792.0,
        "flop_pop": 0.0,
        "iop_pop": 3.2385000506015635
      },
      "hbm": {
        "rd": 8393216.0,
        "wr": 4285952.0,
        "rd_pop": 14.485953229272706,
        "wr_pop": 7.397176507182445
      },
      "lds": {
        "lds": 8388608.0,
        "req": 32768.0,
        "util": 1.5051965062794228,
        "bc": 1.0,
        "ins_cu_lds": 0.0,
        "pop": 1.5240000238125007,
        "peak": 81715.2
      },
      "l1": {
        "l1": 16777216.0,
        "hr": 0.0,
        "util": 61.91622831884054,
        "rd_lat": null,
        "wr_lat": null,
        "coal": 100.0,
        "pop": 3.0480000476250013,
        "peak": 81715.2
      },
      "l2": {
        "l2": 16903168.0,
        "hr": 25.516447567698552,
        "util": 35.22898602268314,
        "rd_lat": 1427.5609681746128,
        "wr_lat": 536.5541672264768,
        "pop": 7.293345633978056,
        "peak": 34406.4
      },
      "atomics": {
        "atomic_lat": 0
      },
      "ai": {
        "hbm": 0.0,
        "l2": 0.0,
        "l1": 0.0
      },
      "wave": {
        "count": 16384.0,
        "cycles": 229157.0,
        "ins_per_wave": 45.0,
        "wave_cycles": 4410.9033203125,
        "dep_wait_cycles": 3614.88330078125,
        "issue_wait_cycles": 168.0166015625,
        "active_cycles": 164.036376953125,
        "occupancy": 0.0,
        "pop": 0.0,
        "max_waves": 9728
      },
      "ipc": {
        "value": 0.27265586516315415
      },
      "cycles": {
        "wave_cycles": 4410.9033203125,
        "active": 164.036376953125,
        "dep_wait": 3614.88330078125,
        "issue_wait": 168.0166015625
      },
      "allocations": {
        "vgpr": 0.0,
        "agpr": 8.0,
        "sgpr": 0.0,
        "lds": 64.0,
        "scratch": 1536.0
      },
      "stalls": {
        "scheduler_pipe": 4.990561028847855,
        "scratch": 0.0,
        "waveslots": 5.180830977061692,
        "vgprs": 0.0,
        "sgprs": 0.0,
        "lds": 0.0,
        "barriers": 0.0,
        "workgroup_limit": 0.0,
        "wavefront_limit": 0.0
      },
      "instruction_mix": {
        "valu": 20,
        "vmem": 2,
        "lds": 2,
        "mfma": 0,
        "salu": 9,
        "smem": 3,
        "branch": 2,
        "compute_mem_ratio": 7.25
      }
    }
  ],
  "initial": [
    {
      "name": "2025_06_04-04_57_29",
      "kernel": "matrix_transpose(float const*, float*, int, int)",
      "count": 1,
      "gpu_series": "MI300",
      "cu_per_gpu": 304,
      "max_sclk": 2100,
      "hbm_bw": 8601.6,
      "lds_banks_per_cu": 32,
      "total_l2_chan": 128,
      "se_per_gpu": 32,
      "num_xcd": 8,
      "grid": 1048576.0,
      "workgroup": 256.0,
      "durations": {
        "ns": 9101.0,
        "pct": 100.0
      },
      "flops": {
        "f": 0.0,
        "i": 11534336.0,
        "flop_pop": 0.0,
        "iop_pop": 1.5509601385069092
      },
      "hbm": {
        "rd": 8392704.0,
        "wr": 4307200.0,
        "rd_pop": 10.720956880719543,
        "wr_pop": 5.502077218097435
      },
      "lds": {
        "lds": 0.0,
        "req": 0.0,
        "util": 0.0,
        "bc": 0,
        "ins_cu_lds": 0.0,
        "pop": 0.0,
        "peak": 81715.2
      },
      "l1": {
        "l1": 142606336.0,
        "hr": 70.58823529411765,
        "util": 66.43869547871758,
        "rd_lat": null,
        "wr_lat": null,
        "coal": 40.0,
        "pop": 19.175507166994514,
        "peak": 81715.2
      },
      "l2": {
        "l2": 42048512.0,
        "hr": 70.06063853103767,
        "util": 36.252319676611904,
        "rd_lat": 1332.204756097561,
        "wr_lat": 536.9240162620927,
        "pop": 13.428338591782168,
        "peak": 34406.4
      },
      "atomics": {
        "atomic_lat": 0
      },
      "ai": {
        "hbm": 0.0,
        "l2": 0.0,
        "l1": 0.0
      },
      "wave": {
        "count": 16384.0,
        "cycles": 278056.0,
        "ins_per_wave": 28.0,
        "wave_cycles": 6316.819580078125,
        "dep_wait_cycles": 5253.550537109375,
        "issue_wait_cycles": 613.336669921875,
        "active_cycles": 100.0,
        "occupancy": 0.0,
        "pop": 0.0,
        "max_waves": 9728
      },
      "ipc": {
        "value": 0.11628423367566587
      },
      "cycles": {
        "wave_cycles": 6316.819580078125,
        "active": 100.0,
        "dep_wait": 5253.550537109375,
        "issue_wait": 613.336669921875
      },
      "allocations": {
        "vgpr": 0.0,
        "agpr": 8.0,
        "sgpr": 0.0,
        "lds": 64.0,
        "scratch": 0.0
      },
      "stalls": {
        "scheduler_pipe": 4.811124865231801,
        "scratch": 0.0,
        "waveslots": 5.432434662915308,
        "vgprs": 0.0,
        "sgprs": 0.0,
        "lds": 0.0,
        "barriers": 0.0,
        "workgroup_limit": 0.0,
        "wavefront_limit": 0.0
      },
      "instruction_mix": {
        "valu": 12,
        "vmem": 2,
        "lds": 0,
        "mfma": 0,
        "salu": 6,
        "smem": 3,
        "branch": 1,
        "compute_mem_ratio": 9.0
      },
      "source": {
        "assembly": [
          "s_load_dword s6, s[0:1], 0x24                              // 000000002100: C0020180 00000024 ",
          "s_load_dwordx2 s[4:5], s[0:1], 0x10                        // 000000002108: C0060100 00000010 ",
          "v_and_b32_e32 v1, 0x3ff, v0                                // 000000002110: 260200FF 000003FF ",
          "v_bfe_u32 v0, v0, 10, 10                                   // 000000002118: D1C80000 02291500 ",
          "s_waitcnt lgkmcnt(0)                                       // 000000002120: BF8CC07F ",
          "s_lshr_b32 s7, s6, 16                                      // 000000002124: 8F079006 ",
          "s_and_b32 s6, s6, 0xffff                                   // 000000002128: 8606FF06 0000FFFF ",
          "s_mul_i32 s2, s2, s6                                       // 000000002130: 92020602 ",
          "s_mul_i32 s3, s3, s7                                       // 000000002134: 92030703 ",
          "v_add_u32_e32 v2, s2, v1                                   // 000000002138: 68040202 ",
          "v_add_u32_e32 v0, s3, v0                                   // 00000000213C: 68000003 ",
          "v_cmp_gt_i32_e32 vcc, s4, v2                               // 000000002140: 7D880404 ",
          "v_cmp_gt_i32_e64 s[2:3], s5, v0                            // 000000002144: D0C40002 00020005 ",
          "s_and_b64 s[2:3], vcc, s[2:3]                              // 00000000214C: 8682026A ",
          "s_and_saveexec_b64 s[6:7], s[2:3]                          // 000000002150: BE862002 ",
          "s_load_dwordx4 s[0:3], s[0:1], 0x0                         // 000000002158: C00A0000 00000000 ",
          "v_mad_u64_u32 v[4:5], s[6:7], v0, s4, v[2:3]               // 000000002160: D1E80604 04080900 ",
          "v_ashrrev_i32_e32 v5, 31, v4                               // 000000002168: 220A089F ",
          "s_waitcnt lgkmcnt(0)                                       // 00000000216C: BF8CC07F ",
          "v_lshl_add_u64 v[4:5], v[4:5], 2, s[0:1]                   // 000000002170: D2080004 00010504 ",
          "global_load_dword v3, v[4:5], off                          // 000000002178: DC508000 037F0004 ",
          "v_mad_u64_u32 v[0:1], s[0:1], v2, s5, v[0:1]               // 000000002180: D1E80000 04000B02 ",
          "v_ashrrev_i32_e32 v1, 31, v0                               // 000000002188: 2202009F ",
          "v_lshl_add_u64 v[0:1], v[0:1], 2, s[2:3]                   // 00000000218C: D2080000 00090500 ",
          "s_waitcnt vmcnt(0)                                         // 000000002194: BF8C0F70 ",
          "global_store_dword v[0:1], v3, off                         // 000000002198: DC708000 007F0300 "
        ],
        "files": [
          "/home/muhaawad/git/amd/audacious/maestro/examples/access_pattern/uncoalesced/uncoalesced.hip",
          "/home/muhaawad/git/amd/audacious/maestro/examples/access_pattern/uncoalesced/uncoalesced.hip",
          "/home/muhaawad/git/amd/audacious/maestro/examples/access_pattern/uncoalesced/uncoalesced.hip",
          "/home/muhaawad/git/amd/audacious/maestro/examples/access_pattern/uncoalesced/uncoalesced.hip",
          "/home/muhaawad/git/amd/audacious/maestro/examples/access_pattern/uncoalesced/uncoalesced.hip",
          "/opt/rocm-6.3.1/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_runtime.h",
          "/opt/rocm-6.3.1/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_runtime.h"
        ],
        "hip": [
          "",
          "  int x = blockIdx.x * blockDim.x + threadIdx.x;  // column",
          "  int y = blockIdx.y * blockDim.y + threadIdx.y;  // row",
          "  if (x < width && y < height) {",
          "    out[x * height + y] = in[y * width + x];",
          "__DEVICE__ unsigned int __hip_get_block_idx_x() { return __ockl_get_group_id(0); }",
          "__DEVICE__ unsigned int __hip_get_block_dim_x() { return __ockl_get_local_size(0); }"
        ],
        "lines": [
          0,
          43,
          44,
          46,
          47,
          270,
          275
        ],
        "signature": "matrix_transpose(float const*, float*, int, int)"
      }
    }
  ],
  "report_message": "The optimized code achieved 100.0% memory coalescing (up from 40.0%, 250.0% improvement), resulting in a 1.351x speedup (from 0.009 ms to 0.007 ms), where higher coalescing percentages indicate more efficient memory access patterns.",
  "bottleneck_report": "Maestro detected uncoalesced memory accesses in the kernel `matrix_transpose` with arguments `float const*, float*, int, int`.",
  "formula": "memoryAccess"
}
```


