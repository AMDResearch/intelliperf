"""Centralized LangChain configuration and LLM management."""
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, Union
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
from langchain_core.output_parsers import PydanticOutputParser

try:
    from langchain_anthropic import ChatAnthropic
    import httpx
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    ChatAnthropic = None

# Callbacks removed for simplified agent architecture


class LLMManager:
    """Centralized LangChain LLM manager for all agents."""

    def __init__(
        self,
        model: str = "openai/gpt-5.2-codex",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        provider: str = "openrouter",
        enable_cache: bool = False,
        enable_callbacks: bool = False,
        cache_path: str = ".intelliperf_cache.db",
        verbose_callbacks: bool = False
    ):
        """Initialize LLM manager.

        Args:
            model: Model name (OpenRouter: provider/model, Azure: deployment-id, Anthropic: claude-sonnet-4.5)
            temperature: Sampling temperature
            max_tokens: Max tokens in response
            provider: LLM provider (openrouter, azure, anthropic)
            enable_cache: Enable SQLite caching for LLM responses
            enable_callbacks: Enable stdout callbacks for debugging
            cache_path: Path to SQLite cache database
            verbose_callbacks: Show detailed callback info
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.provider = provider
        self.enable_callbacks = enable_callbacks
        self.verbose_callbacks = verbose_callbacks

        # Setup LLM cache
        if enable_cache:
            set_llm_cache(SQLiteCache(database_path=cache_path))
            print(f"[LLM] ✓ Cache enabled: {cache_path}")

        # Setup callbacks (disabled in new architecture)
        self.callbacks = []

        # Get API keys for all providers
        self.api_key = self._get_api_key()
        self.azure_api_key = self._get_azure_api_key()
        self.azure_endpoint = self._get_azure_endpoint()
        self.anthropic_api_key = self._get_anthropic_api_key()
        self.anthropic_endpoint = self._get_anthropic_endpoint()
        self.anthropic_version = self._get_anthropic_version()
        
        # Validate provider-specific requirements
        if self.provider == "azure":
            if not self.azure_api_key:
                raise ValueError(
                    "AZURE_OPENAI_API_KEY required for Azure provider. "
                    "Set via: export AZURE_OPENAI_API_KEY='your-subscription-key'"
                )
            if not self.azure_endpoint:
                self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        elif self.provider == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise ValueError(
                    "Anthropic provider requires anthropic package. "
                    "Install via: pip install anthropic httpx"
                )
            if not self.anthropic_api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY required for Anthropic provider. "
                    "Set via: export ANTHROPIC_API_KEY='your-subscription-key'"
                )
        elif not self.api_key:
            raise ValueError(
                "LLM_GATEWAY_KEY or OPENAI_API_KEY environment variable required. "
                "Set via: export LLM_GATEWAY_KEY='your-key'"
            )

        # Initialize LLM
        self.llm = self._create_llm()

        print(f"[LLM] ✓ Initialized: {model} via {provider}")

    def _get_api_key(self) -> Optional[str]:
        """Get API key from environment."""
        return os.getenv("LLM_GATEWAY_KEY") or os.getenv("OPENAI_API_KEY")

    def _get_azure_api_key(self) -> Optional[str]:
        """Get Azure OpenAI API key from environment."""
        return os.getenv("AZURE_OPENAI_API_KEY")

    def _get_azure_endpoint(self) -> Optional[str]:
        """Get Azure OpenAI endpoint from environment."""
        return os.getenv("AZURE_OPENAI_ENDPOINT")

    def _get_anthropic_api_key(self) -> Optional[str]:
        """Get Anthropic API key from environment."""
        return os.getenv("ANTHROPIC_API_KEY")

    def _get_anthropic_endpoint(self) -> str:
        """Get Anthropic endpoint from environment."""
        return os.getenv("ANTHROPIC_ENDPOINT")

    def _get_anthropic_version(self) -> str:
        """Get Anthropic API version from environment."""
        return os.getenv("ANTHROPIC_VERSION")

    def _create_llm(self) -> Union[ChatOpenAI, 'ChatAnthropic']:
        """Create LangChain LLM instance."""
        if self.provider == "openrouter":
            return ChatOpenAI(
                model=self.model,
                openai_api_key=self.api_key,
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        elif self.provider == "openai":
            return ChatOpenAI(
                model=self.model,
                openai_api_key=self.api_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        elif self.provider == "azure":
            # Azure OpenAI via Gateway
            deployment_url = f"{self.azure_endpoint}/openai/deployments/{self.model}"
            return ChatOpenAI(
                model=self.model.split('-')[-1],  # Extract model name from deployment ID
                openai_api_key="dummy",
                openai_api_base=deployment_url,
                temperature=1.0,  # Azure OpenAI requires temperature=1.0
                max_tokens=self.max_tokens,
                default_headers={
                    "Ocp-Apim-Subscription-Key": self.azure_api_key
                }
            )
        elif self.provider == "anthropic":
            # Anthropic API
            headers = {
                'Ocp-Apim-Subscription-Key': self.anthropic_api_key,
                'user': os.getlogin(),
                'anthropic-version': self.anthropic_version
            }
            return ChatAnthropic(
                model=self.model,
                anthropic_api_key="dummy",
                anthropic_api_url=self.anthropic_endpoint,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                default_headers=headers
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def create_chain(self, system_template: str, human_template: str, with_retry: bool = True):
        """Create a LangChain chain with given templates.

        Args:
            system_template: System message template
            human_template: Human message template with {variables}
            with_retry: Add retry logic (3 attempts)

        Returns:
            LangChain chain (prompt | llm) with retries
        """
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])

        # Build chain with LCEL
        chain = prompt | self.llm

        # Add retry logic
        if with_retry:
            chain = chain.with_retry(
                stop_after_attempt=3,
                wait_exponential_jitter=True
            )

        return chain

    def create_structured_chain(self, system_template: str, human_template: str, output_model, with_retry: bool = True):
        """Create a chain with structured Pydantic output.

        Args:
            system_template: System message template
            human_template: Human message template
            output_model: Pydantic model class for output
            with_retry: Add retry logic

        Returns:
            Chain that returns validated Pydantic objects
        """
        parser = PydanticOutputParser(pydantic_object=output_model)

        # Get format instructions and escape ALL curly braces for prompt template
        format_instructions = parser.get_format_instructions()
        # Escape single { and } by doubling them for LangChain template
        escaped_instructions = format_instructions.replace('{', '{{').replace('}', '}}')

        # Add escaped format instructions to prompt
        full_system_template = f"{system_template}\n\n{escaped_instructions}"

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(full_system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])

        # Build chain: prompt → llm → parser
        chain = prompt | self.llm | parser

        if with_retry:
            chain = chain.with_retry(
                stop_after_attempt=3,
                wait_exponential_jitter=True
            )

        return chain

    def invoke(self, messages: list) -> str:
        """Direct invoke with messages.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            LLM response content
        """
        response = self.llm.invoke(messages)
        return response.content

    def get_llm(self) -> Union[ChatOpenAI, 'ChatAnthropic']:
        """Get raw LLM instance for advanced usage."""
        return self.llm

    @staticmethod
    def parse_json_response(response_content: str, context: str = "LLM") -> Optional[Dict[str, Any]]:
        """Parse JSON from LLM response, handling markdown and errors.

        Args:
            response_content: Raw LLM response content
            context: Context string for error messages (e.g., "Tiling", "Algorithmic")

        Returns:
            Parsed JSON dict or None if parsing fails
        """
        try:
            # Clean up response (might have markdown)
            response_text = response_content.strip()

            # Handle markdown code blocks
            if response_text.startswith("```"):
                lines = response_text.split('\n')
                json_lines = []
                in_code = False

                for line in lines:
                    if line.strip().startswith('```'):
                        in_code = not in_code
                        continue
                    if in_code:
                        json_lines.append(line)

                response_text = '\n'.join(json_lines)

            # Check for empty response
            if not response_text:
                print(f"[{context}] ⚠️  LLM returned empty response")
                return None

            # Parse JSON
            return json.loads(response_text)

        except json.JSONDecodeError as e:
            print(f"[{context}] ⚠️  Invalid JSON: {e}")
            print(f"[{context}] 🔍 Raw response: {response_content[:200]}...")
            return None

        except Exception as e:
            print(f"[{context}] ⚠️  JSON parsing error: {e}")
            return None


# Global LLM manager instance (singleton pattern)
_llm_manager: Optional[LLMManager] = None


def get_llm_manager(
    model: str = "openai/gpt-5.2-codex",
    temperature: float = 0.7,
    max_tokens: int = 2048,
    provider: str = "openrouter",
    force_reinit: bool = False,
    enable_callbacks: bool = True
) -> LLMManager:
    """Get or create global LLM manager instance.

    Args:
        model: Model name
        temperature: Sampling temperature
        max_tokens: Max tokens
        provider: LLM provider
        force_reinit: Force re-initialization even if exists
        enable_callbacks: Enable observability callbacks

    Returns:
        LLMManager instance
    """
    global _llm_manager

    if _llm_manager is None or force_reinit:
        _llm_manager = LLMManager(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            provider=provider,
            enable_callbacks=enable_callbacks
        )

    return _llm_manager


def print_llm_stats():
    """Print LLM usage statistics."""
    if _llm_manager and hasattr(_llm_manager, 'callback_handler'):
        _llm_manager.callback_handler.print_stats()

