import textwrap

from akg_agents.op.verifier.profiler_utils import (
    run_profile_scripts_and_collect_results,
)


def test_profile_utils_allows_base_only_package(tmp_path):
    op_name = "toy"
    base_script = tmp_path / f"profile_{op_name}_base.py"
    base_script.write_text(
        textwrap.dedent(
            """
            import json

            with open("base_profile_result.json", "w", encoding="utf-8") as f:
                json.dump({
                    "avg_time_us": 12.5,
                    "per_case_us": [10.0, 15.0],
                    "method": "unit_timer",
                }, f)
            """
        ),
        encoding="utf-8",
    )

    sections = run_profile_scripts_and_collect_results(str(tmp_path), op_name)

    assert sections["base"] == {
        "avg_us": 12.5,
        "per_case_us": [10.0, 15.0],
        "method": "unit_timer",
    }
    assert sections["gen"] is None
