import cv2
import numpy as np


class CourtMap:
    """
    Cria uma representação limpa da quadra de tênis
    vista de cima.

    A quadra é desenhada usando as medidas do CourtModel,
    sem transformar os pixels do vídeo.
    """

    def __init__(
        self,
        court_model,
        pixels_per_meter=50,
        margin=60,
    ):
        self.court_model = court_model
        self.pixels_per_meter = pixels_per_meter
        self.margin = margin

        self.court_width = int(
            court_model.width * pixels_per_meter
        )

        self.court_height = int(
            court_model.length * pixels_per_meter
        )

        self.image_width = (
            self.court_width + (margin * 2)
        )

        self.image_height = (
            self.court_height + (margin * 2)
        )

    def to_pixel(self, point):
        """
        Converte uma posição da quadra em metros
        para uma posição no mapa em pixels.
        """

        x, y = point

        pixel_x = int(
            self.margin
            + (x * self.pixels_per_meter)
        )

        pixel_y = int(
            self.margin
            + (y * self.pixels_per_meter)
        )

        return pixel_x, pixel_y

    def create(self):
        """
        Cria uma imagem limpa da quadra vista de cima.
        """

        # Fundo escuro
        court_image = np.zeros(
            (
                self.image_height,
                self.image_width,
                3,
            ),
            dtype=np.uint8,
        )

        # Cor da área da quadra
        court_image[
            self.margin:
            self.margin + self.court_height,

            self.margin:
            self.margin + self.court_width,
        ] = (50, 90, 50)

        # Desenha todas as linhas oficiais
        for start, end in self.court_model.get_lines():

            start_pixel = self.to_pixel(start)
            end_pixel = self.to_pixel(end)

            cv2.line(
                court_image,
                start_pixel,
                end_pixel,
                (255, 255, 255),
                3,
            )

        return court_image

    def draw_player(
        self,
        court_image,
        position,
        color=(0, 255, 0),
        radius=10,
    ):
        """
        Desenha um jogador no mapa.

        position:
            posição do jogador em metros:
            (x, y)
        """

        player_pixel = self.to_pixel(position)

        cv2.circle(
            court_image,
            player_pixel,
            radius,
            color,
            -1,
        )

        return court_image

    def draw_ball(
        self,
        court_image,
        position,
        color=(0, 165, 255),
        radius=6,
    ):
        """
        Desenha a bola no mapa.
        """

        ball_pixel = self.to_pixel(position)

        cv2.circle(
            court_image,
            ball_pixel,
            radius,
            color,
            -1,
        )

        return court_image