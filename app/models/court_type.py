from enum import Enum


class CourtType(Enum):
    SINGLES = {
        "width": 8.23,
        "length": 23.77,
    }

    DOUBLES = {
        "width": 10.97,
        "length": 23.77,
    }

    @property
    def width(self):
        return self.value["width"]

    @property
    def length(self):
        return self.value["length"]