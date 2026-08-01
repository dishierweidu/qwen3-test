from enum import Enum


class MediaModality(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


__all__ = ["MediaModality"]
