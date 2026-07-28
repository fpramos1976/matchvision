from dataclasses import dataclass


@dataclass(frozen=True)
class CourtModel:
    """
    Modelo geométrico oficial de uma quadra de tênis.
    Todas as medidas estão em metros.
    Origem (0,0) no canto superior esquerdo.
    """

    # Dimensões oficiais
    width: float = 10.97
    length: float = 23.77

    singles_width: float = 8.23

    service_line_distance: float = 6.40
    net_y: float = 11.885

    @property
    def corners(self):
        return [
            (0.0, 0.0),
            (self.width, 0.0),
            (self.width, self.length),
            (0.0, self.length),
        ]

    @property
    def center_service_line(self):
        x = self.width / 2

        return (
            (x, self.service_line_distance),
            (x, self.length - self.service_line_distance),
        )

    @property
    def near_service_line(self):
        return (
            (0.0, self.service_line_distance),
            (self.width, self.service_line_distance),
        )

    @property
    def far_service_line(self):
        return (
            (0.0, self.length - self.service_line_distance),
            (self.width, self.length - self.service_line_distance),
        )

    @property
    def baseline_near(self):
        return (
            (0.0, self.length),
            (self.width, self.length),
        )

    @property
    def baseline_far(self):
        return (
            (0.0, 0.0),
            (self.width, 0.0),
        )

    @property
    def left_sideline(self):
        return (
            (0.0, 0.0),
            (0.0, self.length),
        )

    @property
    def right_sideline(self):
        return (
            (self.width, 0.0),
            (self.width, self.length),
        )

    @property
    def net(self):
        return (
            (0.0, self.net_y),
            (self.width, self.net_y),
        )

    def get_lines(self):
        return [
            self.baseline_far,
            self.baseline_near,
            self.left_sideline,
            self.right_sideline,
            self.near_service_line,
            self.far_service_line,
            self.center_service_line,
            self.net,
        ]