# Third Party
import pydantic


class BaseConfig(pydantic.BaseModel):
    """Base configuration class for all config models."""

    model_config = pydantic.ConfigDict(
        # Forbid extra fields to catch typos in YAML
        extra="forbid",
        # Validate on assignment for immediate feedback
        validate_assignment=True,
        # Use enum values for serialization (e.g. "gpt2" not Architecture.GPT2)
        use_enum_values=True,
    )
