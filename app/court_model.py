from dataclasses import dataclass



@dataclass
class CourtCorners:
    top_left: tuple | None = None
    top_right: tuple | None = None
    bottom_left: tuple | None = None
    bottom_right: tuple | None = None

@dataclass
class CourtModel:
    # Linhas de fundo
    baseline_near: tuple | None = None
    baseline_far: tuple | None = None

    # Linhas laterais
    left_sideline: tuple | None = None
    right_sideline: tuple | None = None

    # Linhas de serviço
    near_service_line: tuple | None = None
    far_service_line: tuple | None = None

    # Linha central de serviço
    center_service_line: tuple | None = None

    # Rede
    net: tuple | None = None