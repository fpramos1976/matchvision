import cv2
import numpy as np


class Homography:

    def __init__(self, image_points, court_model):
        self.image_points = image_points
        self.court_model = court_model
        self.matrix = None

    def compute(self):

        world_points = np.float32(self.court_model.corners)

        image_points = np.float32([
            self.image_points.top_left,
            self.image_points.top_right,
            self.image_points.bottom_right,
            self.image_points.bottom_left,
        ])

        self.matrix = cv2.getPerspectiveTransform(
            world_points,
            image_points,
        )

        return self.matrix

    def transform_point(self, point):
        """
        Converte um ponto da quadra (em metros) para um ponto
        na imagem (em pixels).
        """

        if self.matrix is None:
            raise ValueError("A homografia ainda não foi calculada.")

        world_point = np.array([[[point[0], point[1]]]], dtype=np.float32)

        image_point = cv2.perspectiveTransform(
            world_point,
            self.matrix,
        )

        x = int(image_point[0][0][0])
        y = int(image_point[0][0][1])

        return (x, y)

    def transform(self, frame):
        """
        Temporário.

        Ainda não vamos gerar a Bird's-Eye View.
        Apenas devolvemos o frame original para que o
        restante do pipeline continue funcionando.
        """
        return frame