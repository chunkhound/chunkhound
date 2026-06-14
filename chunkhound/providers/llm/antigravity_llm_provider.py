import asyncio
from typing import Any

from loguru import logger

# Import contracts
from chunkhound.interfaces.llm_provider import LLMProvider, LLMResponse
from chunkhound.core.config.llm_config import DEFAULT_LLM_TIMEOUT

try:
    from google.antigravity import Agent, LocalAgentConfig
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False


class AntigravityLLMProvider(LLMProvider):
    """LLM provider wrapping the official google-antigravity SDK."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-3.5-flash",
        timeout: int = DEFAULT_LLM_TIMEOUT,
        max_retries: int = 3,
        **kwargs: Any,
    ):
        """Initialize the Antigravity LLM provider.

        Args:
            api_key: Google API key
            model: Model name to use
            timeout: Request timeout in seconds
            max_retries: Number of retry attempts
        """
        if not SDK_AVAILABLE:
            raise RuntimeError(
                "google-antigravity SDK is not installed. Please run `uv add google-antigravity` to install it."
            )
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries
        self._extra_kwargs = kwargs
        
        # Usage statistics tracking
        self._requests_made = 0
        self._tokens_used = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0

    @property
    def name(self) -> str:
        """Provider name."""
        return "antigravity-sdk"

    @property
    def model(self) -> str:
        """Model identifier."""
        return self._model

    @property
    def timeout(self) -> int:
        """Request timeout in seconds."""
        return self._timeout

    def _build_agent_config(
        self,
        system: str | None = None,
        response_schema: Any = None,
    ) -> Any:
        """Build the LocalAgentConfig for the SDK agent."""
        import os
        from google.antigravity import CapabilitiesConfig, LocalAgentConfig
        from google.antigravity.types import BuiltinTools
        from google.antigravity.hooks import policy

        capabilities = CapabilitiesConfig(
            enabled_tools=[
                BuiltinTools.LIST_DIR,
                BuiltinTools.SEARCH_DIR,
                BuiltinTools.FIND_FILE,
                BuiltinTools.VIEW_FILE,
                BuiltinTools.FINISH,
            ],
            enable_subagents=False,
        )

        policies = [
            policy.deny_all(),
            policy.allow("list_directory"),
            policy.allow("search_directory"),
            policy.allow("find_file"),
            policy.allow("view_file"),
            policy.allow("finish"),
        ]

        config_kwargs = {
            "model": self._model,
            "api_key": self._api_key,
            "capabilities": capabilities,
            "workspaces": [os.getcwd()],
            "policies": policies,
        }
        if system:
            config_kwargs["system_instructions"] = system
        if response_schema is not None:
            config_kwargs["response_schema"] = response_schema

        return LocalAgentConfig(**config_kwargs)

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_completion_tokens: int = 4096,
        timeout: int | None = None,
    ) -> LLMResponse:
        """Generate a completion for the given prompt.

        Args:
            prompt: User prompt
            system: Optional system message
            max_completion_tokens: Maximum completion tokens to generate (Note: Unsupported by the underlying SDK and ignored)
            timeout: Optional timeout override

        Returns:
            LLMResponse with content and metadata
        """
        if max_completion_tokens != 4096:
            logger.warning(
                "Antigravity SDK does not support limiting output tokens via max_completion_tokens. "
                f"Requested limit of {max_completion_tokens} is ignored."
            )

        request_timeout = timeout if timeout is not None else self._timeout
        config = self._build_agent_config(system=system)

        try:
            logger.debug(f"Starting Antigravity SDK Agent session with model: {self._model}")
            async with Agent(config=config) as agent:
                logger.debug(f"Executing chat prompt (timeout: {request_timeout}s)")
                response = await asyncio.wait_for(agent.chat(prompt), timeout=request_timeout)
                
                content = await response.text()
                
                # Extract thoughts/reasoning tokens
                thoughts = getattr(response, "thoughts", "")
                
                # Retrieve token usage from agent conversation
                prompt_tokens = 0
                completion_tokens = 0
                total_tokens = 0
                try:
                    if hasattr(agent, "conversation") and agent.conversation is not None:
                        total_usage = getattr(agent.conversation, "total_usage", None)
                        if total_usage is not None:
                            total_tokens = getattr(total_usage, "total_token_count", 0) or 0
                            prompt_tokens = getattr(total_usage, "prompt_token_count", 0) or 0
                            candidates = getattr(total_usage, "candidates_token_count", 0) or 0
                            thoughts_tokens = getattr(total_usage, "thoughts_token_count", 0) or 0
                            completion_tokens = candidates + thoughts_tokens
                except Exception as e:
                    logger.warning(f"Failed to extract token count from SDK agent context: {e}")
                    
                if not total_tokens:
                    # Estimate tokens based on prompt + response length + thoughts length
                    prompt_tokens = len(prompt) // 4
                    if system:
                        prompt_tokens += len(system) // 4
                    completion_tokens = len(content + thoughts) // 4
                    total_tokens = prompt_tokens + completion_tokens

                self._requests_made += 1
                self._tokens_used += total_tokens
                self._prompt_tokens += prompt_tokens
                self._completion_tokens += completion_tokens

                logger.debug(f"Antigravity SDK complete. Content length: {len(content)}, tokens: {total_tokens}")
                return LLMResponse(
                    content=content,
                    tokens_used=total_tokens,
                    model=self._model,
                    finish_reason="stop",
                )
        except Exception as e:
            logger.error(f"Antigravity SDK call failed: {e}")
            raise RuntimeError(f"Antigravity SDK call failed: {e}") from e

    def _compile_schema_to_pydantic(self, schema: dict[str, Any], name: str = "DynamicResponseModel") -> Any:
        """Dynamically compile JSON schema to a Pydantic model class using create_model."""
        from pydantic import create_model, Field
        from typing import Any, List, Dict, Union, Optional
        
        type_mapping = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        
        if not isinstance(schema, dict):
            return schema
            
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        
        fields = {}
        for field_name, prop in properties.items():
            prop_type = prop.get("type")
            python_type = Any
            
            if prop_type == "object":
                python_type = self._compile_schema_to_pydantic(prop, name=f"{name}_{field_name}")
            elif prop_type == "array":
                items = prop.get("items")
                if isinstance(items, dict) and items.get("type") == "object":
                    item_type = self._compile_schema_to_pydantic(items, name=f"{name}_{field_name}_item")
                    python_type = List[item_type]
                elif isinstance(items, dict) and items.get("type") in type_mapping:
                    python_type = List[type_mapping[items.get("type")]]
                else:
                    python_type = List
            elif prop_type in type_mapping:
                python_type = type_mapping[prop_type]
                
            description = prop.get("description", "")
            
            if field_name in required:
                fields[field_name] = (python_type, Field(..., description=description))
            else:
                fields[field_name] = (Optional[python_type], Field(default=None, description=description))
                
        return create_model(name, **fields)

    async def complete_structured(
        self,
        prompt: str,
        json_schema: dict[str, Any],
        system: str | None = None,
        max_completion_tokens: int = 4096,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Generate a structured JSON completion conforming to the given schema.

        Args:
            prompt: User prompt
            json_schema: JSON Schema definition for structured output
            system: Optional system message
            max_completion_tokens: Maximum completion tokens to generate (Note: Unsupported by the underlying SDK and ignored)
            timeout: Optional timeout override

        Returns:
            Parsed JSON object conforming to schema
        """
        if max_completion_tokens != 4096:
            logger.warning(
                "Antigravity SDK does not support limiting output tokens via max_completion_tokens. "
                f"Requested limit of {max_completion_tokens} is ignored."
            )

        request_timeout = timeout if timeout is not None else self._timeout

        # Compile JSON schema to Pydantic model class
        pydantic_schema = self._compile_schema_to_pydantic(json_schema)
        config = self._build_agent_config(system=system, response_schema=pydantic_schema)

        try:
            logger.debug("Starting Antigravity SDK Agent session for structured completion")
            async with Agent(config=config) as agent:
                logger.debug(f"Executing chat prompt for structured completion (timeout: {request_timeout}s)")
                response = await asyncio.wait_for(agent.chat(prompt), timeout=request_timeout)
                
                # Retrieve token usage from agent conversation
                prompt_tokens = 0
                completion_tokens = 0
                total_tokens = 0
                try:
                    if hasattr(agent, "conversation") and agent.conversation is not None:
                        total_usage = getattr(agent.conversation, "total_usage", None)
                        if total_usage is not None:
                            total_tokens = getattr(total_usage, "total_token_count", 0) or 0
                            prompt_tokens = getattr(total_usage, "prompt_token_count", 0) or 0
                            candidates = getattr(total_usage, "candidates_token_count", 0) or 0
                            thoughts_tokens = getattr(total_usage, "thoughts_token_count", 0) or 0
                            completion_tokens = candidates + thoughts_tokens
                except Exception as e:
                    logger.warning(f"Failed to extract token count from SDK agent context in complete_structured: {e}")
                
                if not total_tokens:
                    prompt_tokens = len(prompt) // 4
                    if system:
                        prompt_tokens += len(system) // 4
                    thoughts = getattr(response, "thoughts", "")
                    content = ""
                    try:
                        content = await response.text()
                    except Exception:
                        pass
                    completion_tokens = len(content + thoughts) // 4
                    total_tokens = prompt_tokens + completion_tokens

                self._requests_made += 1
                self._tokens_used += total_tokens
                self._prompt_tokens += prompt_tokens
                self._completion_tokens += completion_tokens

                # Extract structured output if supported
                if hasattr(response, "structured_output"):
                    result = await response.structured_output()
                    if isinstance(result, dict):
                        import json
                        from chunkhound.utils.json_extraction import parse_and_validate_structured_json
                        return parse_and_validate_structured_json(json.dumps(result), json_schema)

                # Fallback to parsing text output
                content = await response.text()
                from chunkhound.utils.json_extraction import parse_and_validate_structured_json
                return parse_and_validate_structured_json(content, json_schema)
        except Exception as e:
            logger.error(f"Antigravity SDK structured call failed: {e}")
            raise RuntimeError(f"Antigravity SDK structured call failed: {e}") from e

    async def batch_complete(
        self,
        prompts: list[str],
        system: str | None = None,
        max_completion_tokens: int = 4096,
    ) -> list[LLMResponse]:
        """Generate completions for multiple prompts concurrently."""
        tasks = [
            self.complete(
                prompt=prompt,
                system=system,
                max_completion_tokens=max_completion_tokens,
            )
            for prompt in prompts
        ]
        return list(await asyncio.gather(*tasks))

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return len(text) // 4

    async def health_check(self) -> dict[str, Any]:
        """Perform health check."""
        try:
            response = await self.complete("ping", max_completion_tokens=5)
            return {
                "status": "healthy",
                "provider": self.name,
                "model": self._model,
                "test_response": response.content[:50],
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": self.name,
                "error": str(e),
            }

    def get_usage_stats(self) -> dict[str, Any]:
        """Get usage statistics."""
        return {
            "requests_made": self._requests_made,
            "total_tokens": self._tokens_used,
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
        }

    def get_synthesis_concurrency(self) -> int:
        """Get recommended concurrency for parallel synthesis operations."""
        return 5
