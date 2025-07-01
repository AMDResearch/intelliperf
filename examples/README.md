# Examples


## Bank Conflict Formula

```console
intelliperf -vvv --project_directory=./examples\
    --build_command="./scripts/build_examples.sh -c"\
    --instrument_command="./scripts/build_examples.sh -i -c"\
    --formula=bankConflict -- ./build/bank_conflict/matrix_transpose 1024 1024
```

## Memory Access Forumla

```console
intelliperf -vvv --project_directory=./examples\
    --build_command="./scripts/build_examples.sh -c"\
    --instrument_command="./scripts/build_examples.sh -i -c"\
    --formula=memoryAccess -- ./build/access_pattern/uncoalesced
```

## Atomic Contention Formula

```console
intelliperf -vvv --project_directory=./examples\
    --build_command="./scripts/build_examples.sh -c"\
    --instrument_command="./scripts/build_examples.sh -i -c"\
    --formula=atomicContention -- ./build/contention/reduction
```

## Diagnose Only Formula
### HIP
```console
./scripts/build_examples.sh 
intelliperf -vvv --formula=diagnoseOnly -- ./examples/build/contention/reduction
```
### Triton
```console
intelliperf -vvv --formula=diagnoseOnly -- python ./examples/triton/reduce.py
```
### Python/PyTorch
```
intelliperf -vvv --formula=diagnoseOnly -- python ./examples/torch/add.py
```

## Example output:

### Memory Access Forumla


After running, you will get an output similar to:

```console
{
  "optimized": [
    {
      "name": "2025_06_23-01_29_43",
      "kernel": "void matrix_transpose<__hip_bfloat16>(__hip_bfloat16 const*, __hip_bfloat16*, int, int)",
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
        "ns": 6130.0,
        "pct": 100.0
      },
      "flops": {
        "f": 0.0,
        "i": 17825792.0,
        "flop_pop": 0.0,
        "iop_pop": 3.5586519316235123
      },
      "hbm": {
        "rd": 8402368.0,
        "wr": 2194176.0,
        "rd_pop": 15.935363745824594,
        "wr_pop": 4.161326031228152
      },
      "lds": {
        "lds": 8388608.0,
        "req": 32768.0,
        "util": 1.2894955903144063,
        "bc": 0.5,
        "ins_cu_lds": 0.0,
        "pop": 1.6746597325287116,
        "peak": 81715.2
      },
      "l1": {
        "l1": 16777216.0,
        "hr": 0.0,
        "util": 62.68957297035998,
        "rd_lat": null,
        "wr_lat": null,
        "coal": 100.0,
        "pop": 3.349319465057423,
        "peak": 81715.2
      },
      "l2": {
        "l2": 17230848.0,
        "hr": 39.05627859986926,
        "util": 35.15772591555053,
        "rd_lat": 1358.0295812991437,
        "wr_lat": 356.5035710204548,
        "pop": 8.169715683989747,
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
        "cycles": 200617.0,
        "ins_per_wave": 46.0,
        "wave_cycles": 3473.707763671875,
        "dep_wait_cycles": 3637.552490234375,
        "issue_wait_cycles": 390.940673828125,
        "active_cycles": 168.04296875,
        "occupancy": 0.0,
        "pop": 0.0,
        "max_waves": 9728
      },
      "ipc": {
        "value": 0.28950632530664283
      },
      "cycles": {
        "wave_cycles": 3473.707763671875,
        "active": 168.04296875,
        "dep_wait": 3637.552490234375,
        "issue_wait": 390.940673828125
      },
      "allocations": {
        "vgpr": 0.0,
        "agpr": 8.0,
        "sgpr": 0.0,
        "lds": 64.0,
        "scratch": 1024.0
      },
      "stalls": {
        "scheduler_pipe": 3.409467571847723,
        "scratch": 0.0,
        "waveslots": 6.811709560217256,
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
        "smem": 4,
        "branch": 2,
        "compute_mem_ratio": 7.25
      }
    }
  ],
  "initial": [
    {
      "name": "2025_06_23-01_29_43",
      "kernel": "void matrix_transpose<__hip_bfloat16>(__hip_bfloat16 const*, __hip_bfloat16*, int, int)",
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
        "ns": 8573.0,
        "pct": 100.0
      },
      "flops": {
        "f": 0.0,
        "i": 11534336.0,
        "flop_pop": 0.0,
        "iop_pop": 1.646481770739692
      },
      "hbm": {
        "rd": 8400896.0,
        "wr": 2214624.0,
        "rd_pop": 11.392355845872702,
        "wr_pop": 3.0032254503341056
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
        "util": 70.78021263251436,
        "rd_lat": null,
        "wr_lat": null,
        "coal": 40.0,
        "pop": 20.356501892781644,
        "peak": 81715.2
      },
      "l2": {
        "l2": 42355712.0,
        "hr": 75.2097285013176,
        "util": 33.7709804454101,
        "rd_lat": 1252.3206243914315,
        "wr_lat": 346.12892113956576,
        "pop": 14.359520199074614,
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
        "cycles": 276150.0,
        "ins_per_wave": 28.0,
        "wave_cycles": 6150.702392578125,
        "dep_wait_cycles": 5274.410400390625,
        "issue_wait_cycles": 778.582763671875,
        "active_cycles": 100.0,
        "occupancy": 0.0,
        "pop": 0.0,
        "max_waves": 9728
      },
      "ipc": {
        "value": 0.11651977830478495
      },
      "cycles": {
        "wave_cycles": 6150.702392578125,
        "active": 100.0,
        "dep_wait": 5274.410400390625,
        "issue_wait": 778.582763671875
      },
      "allocations": {
        "vgpr": 0.0,
        "agpr": 8.0,
        "sgpr": 0.0,
        "lds": 64.0,
        "scratch": 0.0
      },
      "stalls": {
        "scheduler_pipe": 4.671401176585302,
        "scratch": 0.0,
        "waveslots": 6.351496612253066,
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
          "s_nop 0                                                    // 000000008C24: BF800000 ",
          "s_nop 0                                                    // 000000008C28: BF800000 ",
          "s_nop 0                                                    // 000000008C2C: BF800000 ",
          "s_nop 0                                                    // 000000008C30: BF800000 ",
          "s_nop 0                                                    // 000000008C34: BF800000 ",
          "s_nop 0                                                    // 000000008C38: BF800000 ",
          "s_nop 0                                                    // 000000008C3C: BF800000 ",
          "s_nop 0                                                    // 000000008C40: BF800000 ",
          "s_nop 0                                                    // 000000008C44: BF800000 ",
          "s_nop 0                                                    // 000000008C48: BF800000 ",
          "s_nop 0                                                    // 000000008C4C: BF800000 ",
          "s_nop 0                                                    // 000000008C50: BF800000 ",
          "s_nop 0                                                    // 000000008C54: BF800000 ",
          "s_nop 0                                                    // 000000008C58: BF800000 ",
          "s_nop 0                                                    // 000000008C5C: BF800000 ",
          "s_nop 0                                                    // 000000008C60: BF800000 ",
          "s_nop 0                                                    // 000000008C64: BF800000 ",
          "s_nop 0                                                    // 000000008C68: BF800000 ",
          "s_nop 0                                                    // 000000008C6C: BF800000 ",
          "s_nop 0                                                    // 000000008C70: BF800000 ",
          "s_nop 0                                                    // 000000008C74: BF800000 ",
          "s_nop 0                                                    // 000000008C78: BF800000 ",
          "s_nop 0                                                    // 000000008C7C: BF800000 ",
          "s_nop 0                                                    // 000000008C80: BF800000 ",
          "s_nop 0                                                    // 000000008C84: BF800000 ",
          "s_nop 0                                                    // 000000008C88: BF800000 ",
          "s_nop 0                                                    // 000000008C8C: BF800000 ",
          "s_nop 0                                                    // 000000008C90: BF800000 ",
          "s_nop 0                                                    // 000000008C94: BF800000 ",
          "s_nop 0                                                    // 000000008C98: BF800000 ",
          "s_nop 0                                                    // 000000008C9C: BF800000 ",
          "s_nop 0                                                    // 000000008CA0: BF800000 ",
          "s_nop 0                                                    // 000000008CA4: BF800000 ",
          "s_nop 0                                                    // 000000008CA8: BF800000 ",
          "s_nop 0                                                    // 000000008CAC: BF800000 ",
          "s_nop 0                                                    // 000000008CB0: BF800000 ",
          "s_nop 0                                                    // 000000008CB4: BF800000 ",
          "s_nop 0                                                    // 000000008CB8: BF800000 ",
          "s_nop 0                                                    // 000000008CBC: BF800000 ",
          "s_nop 0                                                    // 000000008CC0: BF800000 ",
          "s_nop 0                                                    // 000000008CC4: BF800000 ",
          "s_nop 0                                                    // 000000008CC8: BF800000 ",
          "s_nop 0                                                    // 000000008CCC: BF800000 ",
          "s_nop 0                                                    // 000000008CD0: BF800000 ",
          "s_nop 0                                                    // 000000008CD4: BF800000 ",
          "s_nop 0                                                    // 000000008CD8: BF800000 ",
          "s_nop 0                                                    // 000000008CDC: BF800000 ",
          "s_nop 0                                                    // 000000008CE0: BF800000 ",
          "s_nop 0                                                    // 000000008CE4: BF800000 ",
          "s_nop 0                                                    // 000000008CE8: BF800000 ",
          "s_nop 0                                                    // 000000008CEC: BF800000 ",
          "s_nop 0                                                    // 000000008CF0: BF800000 ",
          "s_nop 0                                                    // 000000008CF4: BF800000 ",
          "s_nop 0                                                    // 000000008CF8: BF800000 ",
          "s_nop 0                                                    // 000000008CFC: BF800000 ",
          "s_nop 0                                                    // 000000008D00: BF800000 ",
          "s_nop 0                                                    // 000000008D04: BF800000 ",
          "s_nop 0                                                    // 000000008D08: BF800000 ",
          "s_nop 0                                                    // 000000008D0C: BF800000 ",
          "s_nop 0                                                    // 000000008D10: BF800000 ",
          "s_nop 0                                                    // 000000008D14: BF800000 ",
          "s_nop 0                                                    // 000000008D18: BF800000 ",
          "s_nop 0                                                    // 000000008D1C: BF800000 ",
          "s_nop 0                                                    // 000000008D20: BF800000 ",
          "s_nop 0                                                    // 000000008D24: BF800000 ",
          "s_nop 0                                                    // 000000008D28: BF800000 ",
          "s_nop 0                                                    // 000000008D2C: BF800000 ",
          "s_nop 0                                                    // 000000008D30: BF800000 ",
          "s_nop 0                                                    // 000000008D34: BF800000 ",
          "s_nop 0                                                    // 000000008D38: BF800000 ",
          "s_nop 0                                                    // 000000008D3C: BF800000 ",
          "s_nop 0                                                    // 000000008D40: BF800000 ",
          "s_nop 0                                                    // 000000008D44: BF800000 ",
          "s_nop 0                                                    // 000000008D48: BF800000 ",
          "s_nop 0                                                    // 000000008D4C: BF800000 ",
          "s_nop 0                                                    // 000000008D50: BF800000 ",
          "s_nop 0                                                    // 000000008D54: BF800000 ",
          "s_nop 0                                                    // 000000008D58: BF800000 ",
          "s_nop 0                                                    // 000000008D5C: BF800000 ",
          "s_nop 0                                                    // 000000008D60: BF800000 ",
          "s_nop 0                                                    // 000000008D64: BF800000 ",
          "s_nop 0                                                    // 000000008D68: BF800000 ",
          "s_nop 0                                                    // 000000008D6C: BF800000 ",
          "s_nop 0                                                    // 000000008D70: BF800000 ",
          "s_nop 0                                                    // 000000008D74: BF800000 ",
          "s_nop 0                                                    // 000000008D78: BF800000 ",
          "s_nop 0                                                    // 000000008D7C: BF800000 ",
          "s_nop 0                                                    // 000000008D80: BF800000 ",
          "s_nop 0                                                    // 000000008D84: BF800000 ",
          "s_nop 0                                                    // 000000008D88: BF800000 ",
          "s_nop 0                                                    // 000000008D8C: BF800000 ",
          "s_nop 0                                                    // 000000008D90: BF800000 ",
          "s_nop 0                                                    // 000000008D94: BF800000 ",
          "s_nop 0                                                    // 000000008D98: BF800000 ",
          "s_nop 0                                                    // 000000008D9C: BF800000 ",
          "s_nop 0                                                    // 000000008DA0: BF800000 ",
          "s_nop 0                                                    // 000000008DA4: BF800000 ",
          "s_nop 0                                                    // 000000008DA8: BF800000 ",
          "s_nop 0                                                    // 000000008DAC: BF800000 ",
          "s_nop 0                                                    // 000000008DB0: BF800000 ",
          "s_nop 0                                                    // 000000008DB4: BF800000 ",
          "s_nop 0                                                    // 000000008DB8: BF800000 ",
          "s_nop 0                                                    // 000000008DBC: BF800000 ",
          "s_nop 0                                                    // 000000008DC0: BF800000 ",
          "s_nop 0                                                    // 000000008DC4: BF800000 ",
          "s_nop 0                                                    // 000000008DC8: BF800000 ",
          "s_nop 0                                                    // 000000008DCC: BF800000 ",
          "s_nop 0                                                    // 000000008DD0: BF800000 ",
          "s_nop 0                                                    // 000000008DD4: BF800000 ",
          "s_nop 0                                                    // 000000008DD8: BF800000 ",
          "s_nop 0                                                    // 000000008DDC: BF800000 ",
          "s_nop 0                                                    // 000000008DE0: BF800000 ",
          "s_nop 0                                                    // 000000008DE4: BF800000 ",
          "s_nop 0                                                    // 000000008DE8: BF800000 ",
          "s_nop 0                                                    // 000000008DEC: BF800000 ",
          "s_nop 0                                                    // 000000008DF0: BF800000 ",
          "s_nop 0                                                    // 000000008DF4: BF800000 ",
          "s_nop 0                                                    // 000000008DF8: BF800000 ",
          "s_nop 0                                                    // 000000008DFC: BF800000 ",
          "s_nop 0                                                    // 000000008E00: BF800000 ",
          "s_nop 0                                                    // 000000008E04: BF800000 ",
          "s_nop 0                                                    // 000000008E08: BF800000 ",
          "s_nop 0                                                    // 000000008E0C: BF800000 ",
          "s_nop 0                                                    // 000000008E10: BF800000 ",
          "s_nop 0                                                    // 000000008E14: BF800000 ",
          "s_nop 0                                                    // 000000008E18: BF800000 ",
          "s_nop 0                                                    // 000000008E1C: BF800000 ",
          "s_nop 0                                                    // 000000008E20: BF800000 ",
          "s_nop 0                                                    // 000000008E24: BF800000 ",
          "s_nop 0                                                    // 000000008E28: BF800000 ",
          "s_nop 0                                                    // 000000008E2C: BF800000 ",
          "s_nop 0                                                    // 000000008E30: BF800000 ",
          "s_nop 0                                                    // 000000008E34: BF800000 ",
          "s_nop 0                                                    // 000000008E38: BF800000 ",
          "s_nop 0                                                    // 000000008E3C: BF800000 ",
          "s_nop 0                                                    // 000000008E40: BF800000 ",
          "s_nop 0                                                    // 000000008E44: BF800000 ",
          "s_nop 0                                                    // 000000008E48: BF800000 ",
          "s_nop 0                                                    // 000000008E4C: BF800000 ",
          "s_nop 0                                                    // 000000008E50: BF800000 ",
          "s_nop 0                                                    // 000000008E54: BF800000 ",
          "s_nop 0                                                    // 000000008E58: BF800000 ",
          "s_nop 0                                                    // 000000008E5C: BF800000 ",
          "s_nop 0                                                    // 000000008E60: BF800000 ",
          "s_nop 0                                                    // 000000008E64: BF800000 ",
          "s_nop 0                                                    // 000000008E68: BF800000 ",
          "s_nop 0                                                    // 000000008E6C: BF800000 ",
          "s_nop 0                                                    // 000000008E70: BF800000 ",
          "s_nop 0                                                    // 000000008E74: BF800000 ",
          "s_nop 0                                                    // 000000008E78: BF800000 ",
          "s_nop 0                                                    // 000000008E7C: BF800000 ",
          "s_nop 0                                                    // 000000008E80: BF800000 ",
          "s_nop 0                                                    // 000000008E84: BF800000 ",
          "s_nop 0                                                    // 000000008E88: BF800000 ",
          "s_nop 0                                                    // 000000008E8C: BF800000 ",
          "s_nop 0                                                    // 000000008E90: BF800000 ",
          "s_nop 0                                                    // 000000008E94: BF800000 ",
          "s_nop 0                                                    // 000000008E98: BF800000 ",
          "s_nop 0                                                    // 000000008E9C: BF800000 ",
          "s_nop 0                                                    // 000000008EA0: BF800000 ",
          "s_nop 0                                                    // 000000008EA4: BF800000 ",
          "s_nop 0                                                    // 000000008EA8: BF800000 ",
          "s_nop 0                                                    // 000000008EAC: BF800000 ",
          "s_nop 0                                                    // 000000008EB0: BF800000 ",
          "s_nop 0                                                    // 000000008EB4: BF800000 ",
          "s_nop 0                                                    // 000000008EB8: BF800000 ",
          "s_nop 0                                                    // 000000008EBC: BF800000 ",
          "s_nop 0                                                    // 000000008EC0: BF800000 ",
          "s_nop 0                                                    // 000000008EC4: BF800000 ",
          "s_nop 0                                                    // 000000008EC8: BF800000 ",
          "s_nop 0                                                    // 000000008ECC: BF800000 ",
          "s_nop 0                                                    // 000000008ED0: BF800000 ",
          "s_nop 0                                                    // 000000008ED4: BF800000 ",
          "s_nop 0                                                    // 000000008ED8: BF800000 ",
          "s_nop 0                                                    // 000000008EDC: BF800000 ",
          "s_nop 0                                                    // 000000008EE0: BF800000 ",
          "s_nop 0                                                    // 000000008EE4: BF800000 ",
          "s_nop 0                                                    // 000000008EE8: BF800000 ",
          "s_nop 0                                                    // 000000008EEC: BF800000 ",
          "s_nop 0                                                    // 000000008EF0: BF800000 ",
          "s_nop 0                                                    // 000000008EF4: BF800000 ",
          "s_nop 0                                                    // 000000008EF8: BF800000 ",
          "s_nop 0                                                    // 000000008EFC: BF800000 ",
          "s_nop 0                                                    // 000000008F00: BF800000 ",
          "s_nop 0                                                    // 000000008F04: BF800000 ",
          "s_nop 0                                                    // 000000008F08: BF800000 ",
          "s_nop 0                                                    // 000000008F0C: BF800000 ",
          "s_nop 0                                                    // 000000008F10: BF800000 ",
          "s_nop 0                                                    // 000000008F14: BF800000 ",
          "s_nop 0                                                    // 000000008F18: BF800000 ",
          "s_nop 0                                                    // 000000008F1C: BF800000 ",
          "s_nop 0                                                    // 000000008F20: BF800000 ",
          "s_nop 0                                                    // 000000008F24: BF800000 ",
          "s_nop 0                                                    // 000000008F28: BF800000 ",
          "s_nop 0                                                    // 000000008F2C: BF800000 ",
          "s_nop 0                                                    // 000000008F30: BF800000 ",
          "s_nop 0                                                    // 000000008F34: BF800000 ",
          "s_nop 0                                                    // 000000008F38: BF800000 ",
          "s_nop 0                                                    // 000000008F3C: BF800000 ",
          "s_nop 0                                                    // 000000008F40: BF800000 ",
          "s_nop 0                                                    // 000000008F44: BF800000 ",
          "s_nop 0                                                    // 000000008F48: BF800000 ",
          "s_nop 0                                                    // 000000008F4C: BF800000 ",
          "s_nop 0                                                    // 000000008F50: BF800000 ",
          "s_nop 0                                                    // 000000008F54: BF800000 ",
          "s_nop 0                                                    // 000000008F58: BF800000 ",
          "s_nop 0                                                    // 000000008F5C: BF800000 ",
          "s_nop 0                                                    // 000000008F60: BF800000 ",
          "s_nop 0                                                    // 000000008F64: BF800000 ",
          "s_nop 0                                                    // 000000008F68: BF800000 ",
          "s_nop 0                                                    // 000000008F6C: BF800000 ",
          "s_nop 0                                                    // 000000008F70: BF800000 ",
          "s_nop 0                                                    // 000000008F74: BF800000 ",
          "s_nop 0                                                    // 000000008F78: BF800000 ",
          "s_nop 0                                                    // 000000008F7C: BF800000 ",
          "s_nop 0                                                    // 000000008F80: BF800000 ",
          "s_nop 0                                                    // 000000008F84: BF800000 ",
          "s_nop 0                                                    // 000000008F88: BF800000 ",
          "s_nop 0                                                    // 000000008F8C: BF800000 ",
          "s_nop 0                                                    // 000000008F90: BF800000 ",
          "s_nop 0                                                    // 000000008F94: BF800000 ",
          "s_nop 0                                                    // 000000008F98: BF800000 ",
          "s_nop 0                                                    // 000000008F9C: BF800000 ",
          "s_nop 0                                                    // 000000008FA0: BF800000 ",
          "s_nop 0                                                    // 000000008FA4: BF800000 ",
          "s_nop 0                                                    // 000000008FA8: BF800000 ",
          "s_nop 0                                                    // 000000008FAC: BF800000 ",
          "s_nop 0                                                    // 000000008FB0: BF800000 ",
          "s_nop 0                                                    // 000000008FB4: BF800000 ",
          "s_nop 0                                                    // 000000008FB8: BF800000 ",
          "s_nop 0                                                    // 000000008FBC: BF800000 ",
          "s_nop 0                                                    // 000000008FC0: BF800000 ",
          "s_nop 0                                                    // 000000008FC4: BF800000 ",
          "s_nop 0                                                    // 000000008FC8: BF800000 ",
          "s_nop 0                                                    // 000000008FCC: BF800000 ",
          "s_nop 0                                                    // 000000008FD0: BF800000 ",
          "s_nop 0                                                    // 000000008FD4: BF800000 ",
          "s_nop 0                                                    // 000000008FD8: BF800000 ",
          "s_nop 0                                                    // 000000008FDC: BF800000 ",
          "s_nop 0                                                    // 000000008FE0: BF800000 ",
          "s_nop 0                                                    // 000000008FE4: BF800000 ",
          "s_nop 0                                                    // 000000008FE8: BF800000 ",
          "s_nop 0                                                    // 000000008FEC: BF800000 ",
          "s_nop 0                                                    // 000000008FF0: BF800000 ",
          "s_nop 0                                                    // 000000008FF4: BF800000 ",
          "s_nop 0                                                    // 000000008FF8: BF800000 ",
          "s_nop 0                                                    // 000000008FFC: BF800000 ",
          "s_nop 0                                                    // 000000009000: BF800000 ",
          "s_nop 0                                                    // 000000009004: BF800000 ",
          "s_nop 0                                                    // 000000009008: BF800000 ",
          "s_nop 0                                                    // 00000000900C: BF800000 ",
          "s_nop 0                                                    // 000000009010: BF800000 ",
          "s_nop 0                                                    // 000000009014: BF800000 ",
          "s_nop 0                                                    // 000000009018: BF800000 ",
          "s_nop 0                                                    // 00000000901C: BF800000 ",
          "s_nop 0                                                    // 000000009020: BF800000 ",
          "s_nop 0                                                    // 000000009024: BF800000 ",
          "s_nop 0                                                    // 000000009028: BF800000 ",
          "s_nop 0                                                    // 00000000902C: BF800000 ",
          "s_nop 0                                                    // 000000009030: BF800000 ",
          "s_nop 0                                                    // 000000009034: BF800000 ",
          "s_nop 0                                                    // 000000009038: BF800000 ",
          "s_nop 0                                                    // 00000000903C: BF800000 ",
          "v_cndmask_b32_e32 v0, s0, v0, vcc                          // 000000009040: 00000000 ",
          "v_cndmask_b32_e32 v0, s0, v0, vcc                          // 000000009044: 00000000 ",
          "v_cndmask_b32_e32 v0, s0, v0, vcc                          // 000000009048: 00000000 ",
          "v_cndmask_b32_e32 v0, s0, v0, vcc                          // 00000000904C: 00000000 ",
          "v_cndmask_b32_e32 v0, s0, v0, vcc                          // 000000009050: 00000000 ",
          "v_cndmask_b32_e32 v0, s0, v0, vcc                          // 000000009054: 00000000 ",
          "v_cndmask_b32_e32 v0, s0, v0, vcc                          // 000000009058: 00000000 ",
          "v_cndmask_b32_e32 v0, s0, v0, vcc                          // 00000000905C: 00000000 ",
          "v_cndmask_b32_e32 v0, s0, v0, vcc                          // 000000009060: 00000000 ",
          "v_cndmask_b32_e32 v0, s0, v0, vcc                          // 000000009064: 00000000 ",
          "v_cndmask_b32_e32 v0, s0, v0, vcc                          // 000000009068: 00000000 ",
          "v_cndmask_b32_e32 v0, s0, v0, vcc                          // 00000000906C: 00000000 ",
          "v_cndmask_b32_e32 v0, s0, v0, vcc                          // 000000009070: 00000000 ",
          "v_cndmask_b32_e32 v0, s0, v0, vcc                          // 000000009074: 00000000 ",
          "v_cndmask_b32_e32 v0, s0, v0, vcc                          // 000000009078: 00000000 ",
          "v_cndmask_b32_e32 v0, s0, v0, vcc                          // 00000000907C: 00000000 ",
          "v_cndmask_b32_e32 v0, s0, v0, vcc                          // 000000009080: 00000000 ",
          "v_cndmask_b32_e32 v0, s0, v0, vcc                          // 000000009084: 00000000 ",
          "v_cndmask_b32_e32 v0, s0, v0, vcc                          // 000000009088: 00000000 ",
          "v_cndmask_b32_e32 v0, s0, v0, vcc                          // 00000000908C: 00000000 ",
          "v_cndmask_b32_e32 v0, s0, v0, vcc                          // 000000009090: 00000000 ",
          "v_cndmask_b32_e32 v0, s0, v0, vcc                          // 000000009094: 00000000 ",
          "v_cndmask_b32_e32 v0, s0, v0, vcc                          // 000000009098: 00000000 ",
          "v_cndmask_b32_e32 v0, s0, v0, vcc                          // 00000000909C: 00000000 ",
          "v_cndmask_b32_e32 v0, s0, v0, vcc                          // 0000000090A0: 00000000 ",
          "v_cndmask_b32_e32 v0, s0, v0, vcc                          // 0000000090A4: 00000000 ",
          "v_cndmask_b32_e32 v0, s0, v0, vcc                          // 0000000090A8: 00000000 ",
          "v_cndmask_b32_e32 v0, s0, v0, vcc                          // 0000000090AC: 00000000 ",
          "v_cndmask_b32_e32 v0, s0, v0, vcc                          // 0000000090B0: 00000000 ",
          "v_cndmask_b32_e32 v0, s0, v0, vcc                          // 0000000090B4: 00000000 ",
          "v_cndmask_b32_e32 v0, s0, v0, vcc                          // 0000000090B8: 00000000 ",
          "v_cndmask_b32_e32 v0, s0, v0, vcc                          // 0000000090BC: 00000000 ",
          "v_cndmask_b32_e32 v0, s0, v0, vcc                          // 0000000090C0: 00000000 ",
          "v_cndmask_b32_e32 v0, s0, v0, vcc                          // 0000000090C4: 00000000 ",
          "v_cndmask_b32_e32 v0, s0, v0, vcc                          // 0000000090C8: 00000000 ",
          "v_cndmask_b32_e32 v0, s0, v0, vcc                          // 0000000090CC: 00000000 ",
          "v_cndmask_b32_e32 v0, s0, v0, vcc                          // 0000000090D0: 00000000 ",
          "v_cndmask_b32_e32 v0, s0, v0, vcc                          // 0000000090D4: 00000000 ",
          "v_cndmask_b32_e32 v0, s0, v0, vcc                          // 0000000090D8: 00000000 ",
          "v_cndmask_b32_e32 v0, s0, v0, vcc                          // 0000000090DC: 00000000 ",
          "v_cndmask_b32_e32 v0, s0, v0, vcc                          // 0000000090E0: 00000000 ",
          "v_cndmask_b32_e32 v0, s0, v0, vcc                          // 0000000090E4: 00000000 ",
          "v_cndmask_b32_e32 v0, s0, v0, vcc                          // 0000000090E8: 00000000 ",
          "v_cndmask_b32_e32 v0, s0, v0, vcc                          // 0000000090EC: 00000000 ",
          "v_cndmask_b32_e32 v0, s0, v0, vcc                          // 0000000090F0: 00000000 ",
          "v_cndmask_b32_e32 v0, s0, v0, vcc                          // 0000000090F4: 00000000 ",
          "v_cndmask_b32_e32 v0, s0, v0, vcc                          // 0000000090F8: 00000000 ",
          "v_cndmask_b32_e32 v0, s0, v0, vcc                          // 0000000090FC: 00000000 ",
          "s_load_dword s6, s[0:1], 0x24                              // 000000009100: C0020180 00000024 ",
          "s_load_dwordx2 s[4:5], s[0:1], 0x10                        // 000000009108: C0060100 00000010 ",
          "v_and_b32_e32 v1, 0x3ff, v0                                // 000000009110: 260200FF 000003FF ",
          "v_bfe_u32 v0, v0, 10, 10                                   // 000000009118: D1C80000 02291500 ",
          "s_waitcnt lgkmcnt(0)                                       // 000000009120: BF8CC07F ",
          "s_lshr_b32 s7, s6, 16                                      // 000000009124: 8F079006 ",
          "s_and_b32 s6, s6, 0xffff                                   // 000000009128: 8606FF06 0000FFFF ",
          "s_mul_i32 s2, s2, s6                                       // 000000009130: 92020602 ",
          "s_mul_i32 s3, s3, s7                                       // 000000009134: 92030703 ",
          "v_add_u32_e32 v2, s2, v1                                   // 000000009138: 68040202 ",
          "v_add_u32_e32 v0, s3, v0                                   // 00000000913C: 68000003 ",
          "v_cmp_gt_i32_e32 vcc, s4, v2                               // 000000009140: 7D880404 ",
          "v_cmp_gt_i32_e64 s[2:3], s5, v0                            // 000000009144: D0C40002 00020005 ",
          "s_and_b64 s[2:3], vcc, s[2:3]                              // 00000000914C: 8682026A ",
          "s_and_saveexec_b64 s[6:7], s[2:3]                          // 000000009150: BE862002 ",
          "s_load_dwordx4 s[0:3], s[0:1], 0x0                         // 000000009158: C00A0000 00000000 ",
          "v_mad_u64_u32 v[4:5], s[6:7], v0, s4, v[2:3]               // 000000009160: D1E80604 04080900 ",
          "v_ashrrev_i32_e32 v5, 31, v4                               // 000000009168: 220A089F ",
          "s_waitcnt lgkmcnt(0)                                       // 00000000916C: BF8CC07F ",
          "v_lshl_add_u64 v[4:5], v[4:5], 1, s[0:1]                   // 000000009170: D2080004 00010304 ",
          "global_load_ushort v3, v[4:5], off                         // 000000009178: DC488000 037F0004 ",
          "v_mad_u64_u32 v[0:1], s[0:1], v2, s5, v[0:1]               // 000000009180: D1E80000 04000B02 ",
          "v_ashrrev_i32_e32 v1, 31, v0                               // 000000009188: 2202009F ",
          "v_lshl_add_u64 v[0:1], v[0:1], 1, s[2:3]                   // 00000000918C: D2080000 00090300 ",
          "s_waitcnt vmcnt(0)                                         // 000000009194: BF8C0F70 ",
          "global_store_short v[0:1], v3, off                         // 000000009198: DC688000 007F0300 "
        ],
        "files": [
          "/home/AMD/muhaawad/git/amd/audacious/intelliperf/examples/access_pattern/uncoalesced/uncoalesced.hip",
          "/home/AMD/muhaawad/git/amd/audacious/intelliperf/examples/access_pattern/uncoalesced/uncoalesced.hip",
          "/home/AMD/muhaawad/git/amd/audacious/intelliperf/examples/access_pattern/uncoalesced/uncoalesced.hip",
          "/home/AMD/muhaawad/git/amd/audacious/intelliperf/examples/access_pattern/uncoalesced/uncoalesced.hip",
          "/home/AMD/muhaawad/git/amd/audacious/intelliperf/examples/access_pattern/uncoalesced/uncoalesced.hip",
          "/home/AMD/muhaawad/git/amd/audacious/intelliperf/examples/access_pattern/uncoalesced/uncoalesced.hip",
          "/opt/rocm-6.3.1/lib/llvm/bin/./././include/hip/amd_detail/amd_hip_runtime.h",
          "/opt/rocm-6.3.1/lib/llvm/bin/./././include/hip/amd_detail/amd_hip_runtime.h"
        ],
        "hip": [
          "",
          "  int x = blockIdx.x * blockDim.x + threadIdx.x;  // column",
          "  int y = blockIdx.y * blockDim.y + threadIdx.y;  // row",
          "  if (x < width && y < height) {",
          "    out[x * height + y] = in[y * width + x];",
          "}",
          "__DEVICE__ unsigned int __hip_get_block_idx_x() { return __ockl_get_group_id(0); }",
          "__DEVICE__ unsigned int __hip_get_block_dim_x() { return __ockl_get_local_size(0); }"
        ],
        "lines": [
          0,
          47,
          48,
          50,
          51,
          53,
          270,
          275
        ],
        "signature": "void matrix_transpose<__hip_bfloat16>(__hip_bfloat16 const*, __hip_bfloat16*, int, int)"
      }
    }
  ],
  "report_message": "Memory Coalescing Improvement: Successfully improved memory access patterns by 250.0%. Coalescing efficiency increased from 40.0% to 100.0% (higher percentages indicate more efficient memory access patterns). Performance Gain: Achieved 1.40x speedup with execution time reduced from 0.01ms to 0.01ms (39.9% faster).",
  "bottleneck_report": "Memory Access Pattern Detection: IntelliPerf identified inefficient memory access patterns in kernel `void matrix_transpose` with arguments `__hip_bfloat16 const*, __hip_bfloat16*, int, int`. Uncoalesced memory accesses occur when threads access memory in non-sequential patterns, reducing memory bandwidth utilization.",
  "formula": "memoryAccess"
}
```


