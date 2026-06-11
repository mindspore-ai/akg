# AutoResearch DSL Examples

This directory contains seed kernels that are shaped for direct
`/autoresearch --ref ... --kernel ... --op-name ...` use.

Each DSL subdirectory contains:

- `reference.py`: the PyTorch reference model and input factory.
- A seed kernel entry accepted by the matching DSL adapter.
- `README.md`: the exact `--kernel` shape and a starter command.

Before running an example, set `akg_agents/workspace_autoresearch/config.yaml`
`defaults.dsl`, `defaults.backend`, and `defaults.framework` to the matching
target. The examples use `torch` references and a simple vector add workload.

Directory-backed DSLs pass the project directory as `--kernel`:

- `ascendc/add_custom/ascendc_op`
- `ascendc_catlass/matmul/catlass_op`

Single-file DSLs pass the seed `.py` file as `--kernel`.
