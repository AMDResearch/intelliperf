import sys
from importlib import import_module
from unittest.mock import Mock

import pytest

from intelliperf.__main__ import main


@pytest.mark.parametrize(
	("formula", "module_name", "report", "message"),
	[
		("atomicContention", "atomic_contention", {"atomics": {"atomic_lat": 10}}, "No atomic contention"),
		("memoryAccess", "memory_access", {"l1": {"coal": 100}}, "No uncoalesced memory access"),
		("bankConflict", "bank_conflict", {"lds": {"bc": 0}}, "No source code or bank conflicts"),
	],
)
def test_cli_stops_when_no_kernel_matches(monkeypatch, caplog, formula, module_name, report, message):
	module = import_module(f"intelliperf.formulas.{module_name}")
	formula_class = getattr(module, module_name)
	instance = object.__new__(formula_class)
	instance._initial_profiler_results = [{"kernel": "copy", **report}]
	instance._instrumentation_results = None
	instance.get_llm = Mock()

	# Exercise the formula's real selection with an already collected report.
	optimizer = Mock()
	optimizer.optimize_pass.side_effect = instance.optimize_pass
	monkeypatch.setattr(module, module_name, Mock(return_value=optimizer))
	monkeypatch.setattr("intelliperf.core.llm.validate_llm_credentials", Mock())
	monkeypatch.setattr("intelliperf.utils.env.get_llm_api_key", Mock())
	monkeypatch.setattr(sys, "argv", ["intelliperf", "-f", formula, "--project_directory", ".", "--", "./copy"])

	main()

	optimizer.optimize_pass.assert_called_once_with(target_kernel=None)
	optimizer.build_pass.assert_not_called()
	assert message in caplog.text
