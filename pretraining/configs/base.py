# Third Party
import pydantic


class BaseConfig(pydantic.BaseModel):
    """Base configuration class for all config models."""

    model_config = pydantic.ConfigDict(
        # Forbid extra fields to catch typos in YAML
        extra="forbid",
        # Validate on assignment for immediate feedback
        validate_assignment=True,
        # Keep enum instances for type safety (not just their string values)
        use_enum_values=False,
    )
