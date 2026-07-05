import asyncio
from typing import Any

from loguru import logger

from chunkhound.core.config.llm_config import DEFAULT_LLM_TIMEOUT

# Import contracts
from chunkhound.interfaces.llm_provider import LLMProvider, LLMResponse

try:
    from google.antigravity import Agent  # type: ignore

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
                "google-antigravity SDK is not installed. "
                "Install it with `uv tool install \"chunkhound[antigravity]\"` "
                "or `pip install \"chunkhound[antigravity]\"`."
            )
        # Fail fast when no API key is available, so callers (e.g. MCP tool
        # listing) filter out LLM tools at startup rather than surfacing a
        # deferred SDK error. Auth is the resolved CHUNKHOUND_LLM_API_KEY,
        # matching CLI validation (LLMConfig.get_missing_config_for_roles).
        if not api_key:
            raise ValueError(
                "AntigravityLLMProvider requires an API key: set "
                "CHUNKHOUND_LLM_API_KEY."
            )
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries
        self._target_dir = kwargs.get("target_dir")
        if not self._target_dir:
            raise ValueError(
                "AntigravityLLMProvider requires 'target_dir' to scope SDK "
                "workspaces; refusing to fall back to the process working "
                "directory."
            )
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
        from google.antigravity import CapabilitiesConfig, LocalAgentConfig
        from google.antigravity.hooks import policy  # type: ignore
        from google.antigravity.types import BuiltinTools  # type: ignore

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
            "workspaces": [str(self._target_dir)],
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
            max_completion_tokens: Maximum completion tokens to generate.
                Note: Unsupported by the underlying SDK and ignored.
            timeout: Optional timeout override

        Returns:
            LLMResponse with content and metadata
        """
        if max_completion_tokens is not None and max_completion_tokens != 4096:
            logger.warning(
                "Antigravity SDK does not support limiting output tokens "
                "via max_completion_tokens. "
                f"Requested limit of {max_completion_tokens} is ignored."
            )

        request_timeout = timeout if timeout is not None else self._timeout
        config = self._build_agent_config(system=system)

        try:
            logger.debug(
                f"Starting Antigravity SDK Agent session with model: {self._model}"
            )

            async def _run_session() -> LLMResponse:
                # The entire session lifecycle (__aenter__, chat, text drain,
                # usage extraction, __aexit__) is bounded by the outer
                # ``wait_for`` so a hang in session open/close is also covered
                # by the request timeout. On timeout ``wait_for`` cancels this
                # coroutine and ``async with`` runs ``__aexit__`` on unwind.
                async with Agent(config=config) as agent:
                    logger.debug(
                        f"Executing chat prompt (timeout: {request_timeout}s)"
                    )
                    res = await agent.chat(prompt)
                    content = await res.text()

                    # Extract thoughts/reasoning tokens
                    thoughts = getattr(res, "thoughts", "")

                    # Retrieve token usage from agent conversation
                    prompt_tokens = 0
                    completion_tokens = 0
                    total_tokens = 0
                    try:
                        if (
                            hasattr(agent, "conversation")
                            and agent.conversation is not None
                        ):
                            total_usage = getattr(
                                agent.conversation, "total_usage", None
                            )
                            if total_usage is not None:
                                total_tokens = (
                                    getattr(
                                        total_usage, "total_token_count", 0
                                    )
                                    or 0
                                )
                                prompt_tokens = (
                                    getattr(
                                        total_usage, "prompt_token_count", 0
                                    )
                                    or 0
                                )
                                candidates = (
                                    getattr(
                                        total_usage,
                                        "candidates_token_count",
                                        0,
                                    )
                                    or 0
                                )
                                thoughts_tokens = (
                                    getattr(
                                        total_usage,
                                        "thoughts_token_count",
                                        0,
                                    )
                                    or 0
                                )
                                completion_tokens = candidates + thoughts_tokens
                    except Exception as e:
                        logger.warning(
                            f"Failed to extract token count from SDK agent context: {e}"
                        )

                    if not total_tokens:
                        if prompt_tokens or completion_tokens:
                            # SDK provided component counts but no populated
                            # total; sum them rather than discarding real
                            # numbers for a character estimate.
                            total_tokens = prompt_tokens + completion_tokens
                        else:
                            # No usage data at all; estimate from prompt +
                            # response length + thoughts.
                            prompt_tokens = len(prompt) // 4
                            if system:
                                prompt_tokens += len(system) // 4
                            completion_tokens = len(content + thoughts) // 4
                            total_tokens = prompt_tokens + completion_tokens

                    self._requests_made += 1
                    self._tokens_used += total_tokens
                    self._prompt_tokens += prompt_tokens
                    self._completion_tokens += completion_tokens

                    logger.debug(
                        "Antigravity SDK complete. Content length: "
                        f"{len(content)}, tokens: {total_tokens}"
                    )
                    return LLMResponse(
                        content=content,
                        tokens_used=total_tokens,
                        model=self._model,
                        finish_reason="stop",
                    )

            return await asyncio.wait_for(
                _run_session(), timeout=request_timeout
            )
        except asyncio.TimeoutError as e:
            logger.error(f"Antigravity SDK call failed: timed out after {request_timeout}s")
            raise RuntimeError(f"Antigravity SDK call failed: timed out after {request_timeout}s") from e
        except Exception as e:
            logger.error(f"Antigravity SDK call failed: {e}")
            raise RuntimeError(f"Antigravity SDK call failed: {e}") from e

    @staticmethod
    def _resolve_ref(schema: Any, defs: dict[str, Any]) -> Any:
        """Resolve a ``$ref`` against ``defs``; return ``schema`` unchanged otherwise."""
        if isinstance(schema, dict) and "$ref" in schema:
            ref_name = schema["$ref"].split("/")[-1]
            if ref_name in defs:
                return defs[ref_name]
        return schema

    def _merge_all_of(
        self, schema: dict[str, Any], defs: dict[str, Any]
    ) -> dict[str, Any]:
        """Merge an ``allOf`` schema into a single object schema (intersection).

        Combines ``properties`` and ``required`` from each (ref-resolved)
        subschema plus the parent schema's own object-level keywords. If any
        merged subschema forbids ``additionalProperties``, the result does too.
        Nested ``allOf`` chains are flattened recursively.
        """
        merged: dict[str, Any] = {
            "type": "object",
            "properties": {},
            "required": [],
        }

        def _absorb(sub: Any) -> None:
            sub = self._resolve_ref(sub, defs)
            if not isinstance(sub, dict):
                return
            merged["properties"].update(sub.get("properties", {}))
            for req in sub.get("required", []):
                if req not in merged["required"]:
                    merged["required"].append(req)
            if sub.get("additionalProperties") is False:
                merged["additionalProperties"] = False
            if isinstance(sub.get("allOf"), list):
                for nested in sub["allOf"]:
                    _absorb(nested)

        for sub in schema.get("allOf", []):
            _absorb(sub)
        # Fold in object-level keywords defined on the parent alongside allOf.
        merged["properties"].update(schema.get("properties", {}))
        for req in schema.get("required", []):
            if req not in merged["required"]:
                merged["required"].append(req)
        if schema.get("additionalProperties") is False:
            merged["additionalProperties"] = False

        return merged

    def _compile_schema_to_pydantic(
        self, schema: Any, name: str = "DynamicResponseModel", defs: dict[str, Any] | None = None
    ) -> Any:
        """Dynamically compile a JSON Schema into a Pydantic model class.

        Uses ``create_model`` under the hood.

        Supported subset (covers every schema ChunkHound passes to
        ``complete_structured`` today): nested ``object``/``array`` schemas,
        ``enum``, primitive types, ``$ref`` resolution at the root and inside
        ``properties``/array ``items``, ``allOf`` (merged as a shallow
        intersection), and ``anyOf``/``oneOf`` at the schema root (compiled to a
        ``Union``).

        Known limitations (not exercised by any current caller): (a) scalar
        ``anyOf``/``oneOf`` declared at the property level is not compiled to a
        ``Union`` (falls back to ``Any``); (b) ``allOf`` subschemas are
        shallow-merged (conflicting per-property constraints are last-wins rather
        than deeply intersected); and (c) a nullable primitive type-array
        (``{"type": ["string", "null"]}``) is not narrowed — the compiled field
        falls back to ``Any`` rather than ``str | None``, so the schema handed to
        the SDK is weaker than the caller's (the value is still enforced by the
        final JSON Schema validation against the original schema in
        ``complete_structured``). See ``complete_structured`` for a related
        limitation (d): its post-response optional-``null`` cleanup only
        ``$ref``-resolves the root schema and does not merge a root composition
        keyword, so a schema whose fields live entirely under a bare root
        ``allOf`` is not seen by the cleanup.
        """
        from typing import Any, Literal

        from pydantic import Field, create_model

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

        if defs is None:
            defs = schema.get("$defs", {})

        # Resolve a root-level $ref before reading properties/composition, so a
        # schema shaped as {"$ref": "#/$defs/Model", "$defs": {...}} (a common
        # Pydantic model_json_schema() output) compiles to the referenced model
        # instead of an empty one.
        if isinstance(schema, dict) and "$ref" in schema:
            schema = self._resolve_ref(schema, defs)
            if not isinstance(schema, dict):
                return schema

        # Handle composition logic.
        # allOf is an intersection (must satisfy ALL subschemas): merge them
        # into a single object schema and compile that. anyOf/oneOf are unions
        # (satisfy at least one / exactly one): compile each and join as Union.
        from typing import Union

        if "allOf" in schema:
            return self._compile_schema_to_pydantic(
                self._merge_all_of(schema, defs), name=name, defs=defs
            )

        for comp_key in ("anyOf", "oneOf"):
            if comp_key in schema:
                comp_schemas = schema[comp_key]
                if isinstance(comp_schemas, list) and comp_schemas:
                    types = []
                    for i, s in enumerate(comp_schemas):
                        t = self._compile_schema_to_pydantic(
                            s, name=f"{name}_{comp_key}_{i}", defs=defs
                        )
                        types.append(t)
                    if types:
                        return Union[tuple(types)]

        properties = schema.get("properties", {})
        required = schema.get("required", [])

        fields: dict[str, Any] = {}
        for field_name, prop in properties.items():
            if not isinstance(prop, dict):
                continue

            # Resolve reference if it exists
            if "$ref" in prop:
                ref = prop["$ref"]
                ref_name = ref.split("/")[-1]
                if defs and ref_name in defs:
                    prop = defs[ref_name]

            prop_type = prop.get("type")
            python_type: Any = Any

            # Handle enum first
            if "enum" in prop and isinstance(prop["enum"], list):
                python_type = Literal[tuple(prop["enum"])]

            # Handle object
            elif prop_type == "object":
                python_type = self._compile_schema_to_pydantic(
                    prop, name=f"{name}_{field_name}", defs=defs
                )

            # Handle array
            elif prop_type == "array":
                items = prop.get("items")
                if isinstance(items, dict):
                    if "$ref" in items:
                        ref = items["$ref"]
                        ref_name = ref.split("/")[-1]
                        if defs and ref_name in defs:
                            items = defs[ref_name]
                    items_type = items.get("type") if isinstance(items, dict) else None
                    if items_type == "object":
                        item_type = self._compile_schema_to_pydantic(
                            items, name=f"{name}_{field_name}_item", defs=defs
                        )
                        python_type = list[item_type]  # type: ignore[valid-type]
                    elif isinstance(items, dict) and "enum" in items and isinstance(items["enum"], list):
                        python_type = list[Literal[tuple(items["enum"])]]  # type: ignore[misc]
                    elif isinstance(items_type, str) and items_type in type_mapping:
                        python_type = list[type_mapping[items_type]]  # type: ignore[valid-type]
                    else:
                        python_type = list
                else:
                    python_type = list

            # Handle primitive types
            elif isinstance(prop_type, str) and prop_type in type_mapping:
                python_type = type_mapping[prop_type]

            description = prop.get("description", "")

            # Collect field arguments
            field_kwargs: dict[str, Any] = {}
            if description:
                field_kwargs["description"] = description

            # Numeric bounds
            if "minimum" in prop:
                field_kwargs["ge"] = prop["minimum"]
            if "maximum" in prop:
                field_kwargs["le"] = prop["maximum"]
            if "exclusiveMinimum" in prop:
                field_kwargs["gt"] = prop["exclusiveMinimum"]
            if "exclusiveMaximum" in prop:
                field_kwargs["lt"] = prop["exclusiveMaximum"]

            # Length/Size bounds
            if "minLength" in prop:
                field_kwargs["min_length"] = prop["minLength"]
            elif "minItems" in prop:
                field_kwargs["min_length"] = prop["minItems"]

            if "maxLength" in prop:
                field_kwargs["max_length"] = prop["maxLength"]
            elif "maxItems" in prop:
                field_kwargs["max_length"] = prop["maxItems"]

            # Regex pattern
            if "pattern" in prop:
                field_kwargs["pattern"] = prop["pattern"]

            if field_name in required:
                fields[field_name] = (python_type, Field(..., **field_kwargs))
            else:
                fields[field_name] = (
                    python_type | None,
                    Field(default=None, **field_kwargs),
                )

        model_kwargs: dict[str, Any] = {}
        if schema.get("additionalProperties") is False:
            model_kwargs["__config__"] = {"extra": "forbid"}

        return create_model(name, **model_kwargs, **fields)

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
            max_completion_tokens: Maximum completion tokens to generate.
                Note: Unsupported by the underlying SDK and ignored.
            timeout: Optional timeout override

        Returns:
            Parsed JSON object conforming to schema
        """
        if max_completion_tokens is not None and max_completion_tokens != 4096:
            logger.warning(
                "Antigravity SDK does not support limiting output tokens "
                "via max_completion_tokens. "
                f"Requested limit of {max_completion_tokens} is ignored."
            )

        request_timeout = timeout if timeout is not None else self._timeout

        # Compile JSON schema to Pydantic model class
        pydantic_schema = self._compile_schema_to_pydantic(json_schema)
        config = self._build_agent_config(
            system=system, response_schema=pydantic_schema
        )

        try:
            logger.debug(
                "Starting Antigravity SDK Agent session for structured completion"
            )

            async def _run_session() -> dict[str, Any]:
                # Entire session lifecycle bounded by the outer wait_for
                # (see complete() for rationale).
                async with Agent(config=config) as agent:
                    logger.debug(
                        "Executing chat prompt for structured completion "
                        f"(timeout: {request_timeout}s)"
                    )
                    res = await agent.chat(prompt)
                    structured_val = None
                    try:
                        if hasattr(res, "structured_output"):
                            structured_val = await res.structured_output()
                    except Exception as e:
                        logger.warning(
                            f"structured_output() failed: {e}. Falling back to text."
                        )
                    text_content = ""
                    try:
                        text_content = await res.text()
                    except Exception:
                        pass

                    # Retrieve token usage from agent conversation
                    prompt_tokens = 0
                    completion_tokens = 0
                    total_tokens = 0
                    try:
                        if (
                            hasattr(agent, "conversation")
                            and agent.conversation is not None
                        ):
                            total_usage = getattr(
                                agent.conversation, "total_usage", None
                            )
                            if total_usage is not None:
                                total_tokens = (
                                    getattr(
                                        total_usage, "total_token_count", 0
                                    )
                                    or 0
                                )
                                prompt_tokens = (
                                    getattr(
                                        total_usage, "prompt_token_count", 0
                                    )
                                    or 0
                                )
                                candidates = (
                                    getattr(
                                        total_usage,
                                        "candidates_token_count",
                                        0,
                                    )
                                    or 0
                                )
                                thoughts_tokens = (
                                    getattr(
                                        total_usage,
                                        "thoughts_token_count",
                                        0,
                                    )
                                    or 0
                                )
                                completion_tokens = candidates + thoughts_tokens
                    except Exception as e:
                        logger.warning(
                            "Failed to extract token count from SDK agent context "
                            f"in complete_structured: {e}"
                        )

                    if not total_tokens:
                        if prompt_tokens or completion_tokens:
                            # SDK provided component counts but no populated
                            # total; sum them rather than discarding real
                            # numbers for a character estimate.
                            total_tokens = prompt_tokens + completion_tokens
                        else:
                            # No usage data at all; estimate from length.
                            prompt_tokens = len(prompt) // 4
                            if system:
                                prompt_tokens += len(system) // 4
                            thoughts = getattr(res, "thoughts", "")
                            content = text_content
                            completion_tokens = len(content + thoughts) // 4
                            total_tokens = prompt_tokens + completion_tokens

                    self._requests_made += 1
                    self._tokens_used += total_tokens
                    self._prompt_tokens += prompt_tokens
                    self._completion_tokens += completion_tokens

                    import json

                    from chunkhound.utils.json_extraction import (
                        extract_json_from_response,
                        parse_and_validate_structured_json,
                    )

                    # Drop None only for OPTIONAL keys, so an optional field
                    # validates as "absent" instead of failing on null, while
                    # a required field that legitimately holds null is kept
                    # (dropping it would fail the original schema's `required`).
                    # Recurses into nested objects and object arrays, deciding
                    # required-ness from each level's schema. Ambiguous shapes
                    # (anyOf/oneOf/allOf, or a non-object subschema) are left
                    # untouched to avoid dropping a field a composition branch
                    # requires. Applied uniformly to every return shape below
                    # (raw dict, pydantic model, plain object, and text
                    # fallback) so the same payload validates consistently.
                    defs = json_schema.get("$defs", {})

                    def _object_schema(schema: Any) -> dict[str, Any] | None:
                        resolved = self._resolve_ref(schema, defs)
                        if (
                            isinstance(resolved, dict)
                            and resolved.get("type") == "object"
                            and isinstance(resolved.get("properties"), dict)
                        ):
                            return resolved
                        return None

                    def _schema_permits_null(schema: Any) -> bool:
                        # True only when the (ref-resolved) property schema
                        # affirmatively allows null, so an explicit null on an
                        # optional field is preserved instead of dropped. Returns
                        # False for unknown/non-nullable schemas, keeping the
                        # drop-the-unfilled-null workaround for those.
                        resolved = (
                            self._resolve_ref(schema, defs)
                            if schema is not None
                            else None
                        )
                        if not isinstance(resolved, dict):
                            return False
                        t = resolved.get("type")
                        if t == "null" or (isinstance(t, list) and "null" in t):
                            return True
                        for comp_key in ("anyOf", "oneOf"):
                            for branch in resolved.get(comp_key, []) or []:
                                if _schema_permits_null(branch):
                                    return True
                        return False

                    def _drop_optional_none(
                        data: dict[str, Any], schema: Any
                    ) -> dict[str, Any]:
                        required_keys = set(schema.get("required", []))
                        props = schema.get("properties", {})
                        if not isinstance(props, dict):
                            props = {}
                        cleaned: dict[str, Any] = {}
                        for k, v in data.items():
                            sub = props.get(k)
                            if (
                                v is None
                                and k not in required_keys
                                and not _schema_permits_null(sub)
                            ):
                                continue
                            if isinstance(v, dict):
                                sub_obj = _object_schema(sub)
                                cleaned[k] = (
                                    _drop_optional_none(v, sub_obj)
                                    if sub_obj is not None
                                    else v
                                )
                            elif isinstance(v, list):
                                resolved_sub = (
                                    self._resolve_ref(sub, defs)
                                    if sub is not None
                                    else None
                                )
                                item_obj = None
                                if (
                                    isinstance(resolved_sub, dict)
                                    and resolved_sub.get("type") == "array"
                                ):
                                    item_obj = _object_schema(
                                        resolved_sub.get("items")
                                    )
                                if item_obj is not None:
                                    cleaned[k] = [
                                        _drop_optional_none(item, item_obj)
                                        if isinstance(item, dict)
                                        else item
                                        for item in v
                                    ]
                                else:
                                    cleaned[k] = v
                            else:
                                cleaned[k] = v
                        return cleaned

                    # Limitation (d): the root schema is only ``$ref``-resolved
                    # here, not ``allOf``-merged (unlike
                    # ``_compile_schema_to_pydantic``). A schema whose fields
                    # live entirely under a bare root ``allOf`` therefore has no
                    # top-level ``properties``/``required`` for the cleanup to
                    # read, so its optional nulls may be dropped. No current
                    # caller emits root-composition schemas.
                    root_schema = self._resolve_ref(json_schema, defs)

                    def _validate_text(text: str) -> dict[str, Any]:
                        # Normalize optional-null the same way as the
                        # structured-object branches: parse the model text,
                        # drop optional None keys, then validate. Fall back to
                        # strict validation if the text is not a JSON object we
                        # can clean.
                        try:
                            parsed = json.loads(extract_json_from_response(text))
                        except (ValueError, TypeError):
                            return parse_and_validate_structured_json(
                                text, json_schema
                            )
                        if isinstance(parsed, dict):
                            cleaned = _drop_optional_none(parsed, root_schema)
                            return parse_and_validate_structured_json(
                                json.dumps(cleaned, default=str), json_schema
                            )
                        return parse_and_validate_structured_json(text, json_schema)

                    # Extract structured output if supported
                    if hasattr(res, "structured_output") and structured_val is not None:
                        result = structured_val
                        result_dict = None

                        if isinstance(result, dict):
                            # The real SDK returns a raw dict (json.loads of the
                            # model JSON), so this is the production path.
                            result_dict = _drop_optional_none(result, root_schema)
                        elif hasattr(result, "model_dump"):
                            result_dict = _drop_optional_none(
                                result.model_dump(), root_schema
                            )
                        elif hasattr(result, "dict"):
                            result_dict = _drop_optional_none(result.dict(), root_schema)
                        elif hasattr(result, "__dict__"):
                            result_dict = _drop_optional_none(
                                dict(result.__dict__), root_schema
                            )
                        else:
                            try:
                                result_dict = _drop_optional_none(
                                    dict(result), root_schema
                                )
                            except (TypeError, ValueError):
                                pass

                        if result_dict is not None:
                            try:
                                return parse_and_validate_structured_json(
                                    json.dumps(result_dict, default=str),
                                    json_schema,
                                )
                            except Exception as structured_err:
                                # Structured payload failed schema validation;
                                # fall back to the text output if we have one
                                # before surfacing the error.
                                logger.warning(
                                    f"Structured result failed schema validation: "
                                    f"{structured_err}; attempting text fallback."
                                )
                                if text_content:
                                    try:
                                        return _validate_text(text_content)
                                    except Exception:
                                        pass
                                raise

                    # Fallback to parsing text output
                    return _validate_text(text_content)

            return await asyncio.wait_for(
                _run_session(), timeout=request_timeout
            )
        except asyncio.TimeoutError as e:
            logger.error(f"Antigravity SDK structured call failed: timed out after {request_timeout}s")
            raise RuntimeError(f"Antigravity SDK structured call failed: timed out after {request_timeout}s") from e
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
            response = await self.complete("ping", max_completion_tokens=4096)
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
