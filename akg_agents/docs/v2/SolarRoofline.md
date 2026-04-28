# Solar Roofline

## Overview

AKG Agents' roofline support is provided through an **installed `solar` Python package**.

Runtime constraints:
- AKG does **not** depend on a local `SOLAR` source tree path
- AKG does **not** require modifying or patching the `SOLAR` repository
- AKG maintains the missing bridge pieces internally:
  - SOLBench workload wrapper generation
  - roofline architecture configs for Ascend / A100 / V100

In other words, AKG now only requires:

```bash
python -c "import solar"
```

to succeed.

## Installation

Recommended:

```bash
bash download.sh --with_solar
```

or:

```bash
SOLAR_DIR=/path/to/SOLAR bash download.sh --with_solar
SOLAR_REF=<tag|branch|commit> bash download.sh --with_solar
```

The script will:
1. Clone the official Solar repo into `thirdparty/SOLAR` by default
2. Optionally override the directory with `SOLAR_DIR` and check out a specific `SOLAR_REF`
3. Prefer Solar's own `install.sh`
4. Install Solar dependencies, including patched torchview
5. Install Solar into the current Python environment in editable mode
6. Verify `import solar` and the core APIs used by AKG

## Runtime behavior

When roofline is enabled during profiling, AKG calls Solar's Python APIs directly:

- `solar.graph.PyTorchProcessor`
- `solar.einsum.PyTorchToEinsum`
- `solar.analysis.EinsumGraphAnalyzer`
- `solar.perf.EinsumGraphPerfModel`

Outputs include:
- `roofline_profile_result.json`
- profile return fields:
  - `roofline_time`
  - `roofline_speedup`
  - `roofline`

Where:

```text
roofline_speedup = roofline_time / gen_time
```

`1.0x` means the generated kernel is on the roofline bound; values below `1.0x` mean it is still below the theoretical roofline.

## Graceful degradation

If Solar is not installed, or Solar analysis fails:
- the main correctness / profiling flow still continues
- roofline degrades to missing data only
- AKG profiling does not fail solely because roofline is unavailable
