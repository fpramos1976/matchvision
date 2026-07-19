from dataclasses import dataclass

@dataclass
class Line:
    start: tuple[int, int]
    end: tuple[int, int]


@dataclass
class CourtModel:
    baseline_near: Line | None = None
    baseline_far: Line | None = None

    left_sideline: Line | None = None
    right_sideline: Line | None = None

    service_line_near: Line | None = None
    service_line_far: Line | None = None

    center_service_line: Line | None = None

    net: Line | None = None