"""Architecture-specific configuration profiles and lazy factory registry."""

from .registry import (
    ProfileBuildRequest,
    ProfileBuildResult,
    ProfileFactory,
    get_profile_factory,
    parse_profile,
    register_profile_factory,
)

__all__ = [
    "ProfileBuildRequest",
    "ProfileBuildResult",
    "ProfileFactory",
    "get_profile_factory",
    "parse_profile",
    "register_profile_factory",
]
