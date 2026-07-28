from dataclasses import dataclass


@dataclass
class CourtCorners:
    top_left: tuple | None = None
    top_right: tuple | None = None
    bottom_right: tuple | None = None
    bottom_left: tuple | None = None