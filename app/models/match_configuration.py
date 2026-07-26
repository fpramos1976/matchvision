from dataclasses import dataclass

from app.models.court_type import CourtType


@dataclass
class MatchConfiguration:
    court_type: CourtType