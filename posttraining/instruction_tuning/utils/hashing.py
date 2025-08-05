# Standard Library
import hashlib
import os
import typing

# Third Party
from transformers.utils.hub import _CACHED_NO_EXIST
from transformers.utils.hub import TRANSFORMERS_CACHE
from transformers.utils.hub import extract_commit_hash
from transformers.utils.hub import try_to_load_from_cache


def custom_cached_file(
    model_name_or_path: str, filename: str, revision: str = None, repo_type: str = "model"
) -> typing.Optional[str]:
    """
    Get cached file path, handling both local and HF hub files.

    HF's `cached_file` no longer works for `repo_type="dataset"`, so we use this custom version.
    """
    if os.path.isdir(model_name_or_path):
        resolved_file = os.path.join(model_name_or_path, filename)
        if os.path.isfile(resolved_file):
            return resolved_file
        else:
            return None
    else:
        resolved_file = try_to_load_from_cache(
            model_name_or_path,
            filename,
            cache_dir=TRANSFORMERS_CACHE,
            revision=revision,
            repo_type=repo_type,
        )
        # Special return value from try_to_load_from_cache
        if resolved_file == _CACHED_NO_EXIST:
            return None
        return resolved_file


def get_commit_hash(
    model_name_or_path: str, revision: str, filename: str = "config.json", repo_type: str = "model"
) -> str:
    """Extract commit hash from a cached file."""
    file = custom_cached_file(model_name_or_path, filename, revision=revision, repo_type=repo_type)
    commit_hash = extract_commit_hash(file, None)
    return commit_hash


def get_file_hash(
    model_name_or_path: str, revision: str, filename: str = "config.json", repo_type: str = "model"
) -> str:
    """Get SHA256 hash of a file's contents."""
    file = custom_cached_file(model_name_or_path, filename, revision=revision, repo_type=repo_type)
    if isinstance(file, str):
        with open(file, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    elif file is _CACHED_NO_EXIST:
        return f"{filename} not found"
    elif file is None:
        return f"{filename} not found"
    else:
        raise ValueError(f"Unexpected file type: {type(file)}")


def get_files_hash_if_exists(
    model_name_or_path: str, revision: str, filenames: typing.List[str], repo_type: str = "model"
) -> typing.List[str]:
    """Get hashes for multiple files."""
    return [
        get_file_hash(model_name_or_path, revision, filename, repo_type) for filename in filenames
    ]
