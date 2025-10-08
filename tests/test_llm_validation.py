from unittest.mock import Mock, patch

import pytest
from intelliperf.core.llm import LLM


def test_validate_credentials_amd_success():
	"""Test successful credential validation for AMD provider."""
	with patch("requests.post") as mock_post:
		mock_response = Mock()
		mock_response.status_code = 200
		mock_response.json.return_value = {"choices": [{"message": {"content": "test"}}]}
		mock_post.return_value = mock_response

		# Should not raise an exception
		llm = LLM(
			api_key="test-key", system_prompt="test", model="test-model", provider="https://llm-api.amd.com/azure"
		)
		assert llm is not None


def test_validate_credentials_amd_auth_failure():
	"""Test credential validation fails with 401 for AMD provider."""
	with patch("requests.post") as mock_post:
		mock_response = Mock()
		mock_response.status_code = 401
		mock_response.raise_for_status.side_effect = Exception("401 Unauthorized")
		mock_post.return_value = mock_response

		with pytest.raises(SystemExit):
			LLM(api_key="bad-key", system_prompt="test", model="test-model", provider="https://llm-api.amd.com/azure")


def test_validate_credentials_amd_model_not_found():
	"""Test credential validation fails with 404 for AMD provider."""
	with patch("requests.post") as mock_post:
		mock_response = Mock()
		mock_response.status_code = 404
		mock_response.raise_for_status.side_effect = Exception("404 Not Found")
		mock_post.return_value = mock_response

		with pytest.raises(SystemExit):
			LLM(
				api_key="test-key",
				system_prompt="test",
				model="nonexistent-model",
				provider="https://llm-api.amd.com/azure",
			)


def test_validate_credentials_openai_success():
	"""Test successful credential validation for OpenAI provider."""
	with patch("dspy.LM"), patch("dspy.configure"), patch("dspy.ChainOfThought") as mock_chain:
		# Mock the chain of thought call
		mock_chain_instance = Mock()
		mock_chain_instance.return_value = Mock(output="test")
		mock_chain.return_value = mock_chain_instance

		# Should not raise an exception
		llm = LLM(api_key="test-key", system_prompt="test", model="gpt-4", provider="openai")
		assert llm is not None


def test_validate_credentials_openai_failure():
	"""Test credential validation fails for OpenAI provider."""
	with patch("dspy.LM"), patch("dspy.configure"), patch("dspy.ChainOfThought") as mock_chain:
		# Mock the chain of thought to raise an exception
		mock_chain_instance = Mock()
		mock_chain_instance.side_effect = Exception("Authentication failed")
		mock_chain.return_value = mock_chain_instance

		with pytest.raises(SystemExit):
			LLM(api_key="bad-key", system_prompt="test", model="gpt-4", provider="openai")
