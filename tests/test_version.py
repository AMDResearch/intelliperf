"""Test dynamic versioning with setuptools-scm."""

import re


def test_version_format():
	"""Test that the package version follows the expected format."""
	try:
		from importlib.metadata import version

		pkg_version = version("intelliperf")
		# Version should match pattern: X.Y.Z or X.Y.Z.postN+hash.date
		# Examples: "0.0.0", "0.0.0.post2+g07ded6f.d20251016", "1.2.3"
		pattern = r"^\d+\.\d+\.\d+(\.post\d+(\+g[0-9a-f]+\.d\d{8})?)?$"
		assert re.match(pattern, pkg_version), f"Version {pkg_version} doesn't match expected format"
	except ImportError:
		# Package not installed, skip test
		pass


def test_fallback_version():
	"""Test that fallback version is set correctly in config."""
	import tomllib

	with open("pyproject.toml", "rb") as f:
		config = tomllib.load(f)

	assert "setuptools_scm" in config["tool"], "setuptools_scm configuration missing"
	assert config["tool"]["setuptools_scm"]["fallback_version"] == "0.0.0", "Fallback version should be 0.0.0"


def test_dynamic_version_config():
	"""Test that version is configured as dynamic."""
	import tomllib

	with open("pyproject.toml", "rb") as f:
		config = tomllib.load(f)

	assert "version" in config["project"].get("dynamic", []), "Version should be in dynamic list"
	assert "version" not in config["project"], "Static version should not be set"


def test_build_requirements():
	"""Test that setuptools-scm is in build requirements."""
	import tomllib

	with open("pyproject.toml", "rb") as f:
		config = tomllib.load(f)

	build_reqs = config["build-system"]["requires"]
	assert any("setuptools-scm" in req for req in build_reqs), "setuptools-scm should be in build requirements"
