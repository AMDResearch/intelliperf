from unittest.mock import Mock, patch

import pytest

from intelliperf.core.llm import validate_llm_credentials


def test_validate_credentials_amd_success():
	"""Test successful credential validation for AMD provider."""
	with patch("requests.post") as mock_post, patch("dspy.LM"), patch("dspy.configure"):
		mock_response = Mock()
		mock_response.status_code = 200
		mock_response.json.return_value = {"choices": [{"message": {"content": "test"}}]}
		mock_post.return_value = mock_response

		# Should not raise an exception
		result = validate_llm_credentials(
			api_key="test-key", model="test-model", provider="https://llm-api.amd.com/azure"
		)
		assert result is True


def test_validate_credentials_amd_auth_failure():
	"""Test credential validation fails with 401 for AMD provider."""
	with patch("requests.post") as mock_post, patch("dspy.LM"), patch("dspy.configure"):
		mock_response = Mock()
		mock_response.status_code = 401
		mock_http_error = Exception("401 Unauthorized")
		mock_http_error.response = mock_response
		mock_response.raise_for_status.side_effect = mock_http_error
		mock_post.return_value = mock_response

		with pytest.raises(SystemExit):
			validate_llm_credentials(api_key="bad-key", model="test-model", provider="https://llm-api.amd.com/azure")


def test_validate_credentials_amd_model_not_found():
	"""Test credential validation fails with 404 for AMD provider."""
	with patch("requests.post") as mock_post, patch("dspy.LM"), patch("dspy.configure"):
		mock_response = Mock()
		mock_response.status_code = 404
		mock_http_error = Exception("404 Not Found")
		mock_http_error.response = mock_response
		mock_response.raise_for_status.side_effect = mock_http_error
		mock_post.return_value = mock_response

		with pytest.raises(SystemExit):
			validate_llm_credentials(
				api_key="test-key",
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
		result = validate_llm_credentials(api_key="test-key", model="gpt-4", provider="openai")
		assert result is True


def test_validate_credentials_openai_failure():
	"""Test credential validation fails for OpenAI provider."""
	with patch("dspy.LM"), patch("dspy.configure"), patch("dspy.ChainOfThought") as mock_chain:
		# Mock the chain of thought to raise an exception
		mock_chain_instance = Mock()
		mock_chain_instance.side_effect = Exception("Authentication failed")
		mock_chain.return_value = mock_chain_instance

		with pytest.raises(SystemExit):
			validate_llm_credentials(api_key="bad-key", model="gpt-4", provider="openai")
