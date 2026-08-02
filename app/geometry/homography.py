import cv2
import numpy as np


class Homography:

    def __init__(self, image_points, court_model):
        self.image_points = image_points
        self.court_model = court_model
        self.matrix = None
        self.inverse_matrix = None

    def compute(self):
        # Definindo explicitamente as coordenadas do mundo (metros)
        # na MESMA ordem exata de image_points
        world_points = np.float32([
            [0.0, 0.0],                         # top_left
            [self.court_model.width, 0.0],      # top_right
            [self.court_model.width, self.court_model.length], # bottom_right
            [0.0, self.court_model.length]      # bottom_left
        ])

        image_points = np.float32([
            self.image_points.top_left,
            self.image_points.top_right,
            self.image_points.bottom_right,
            self.image_points.bottom_left,
        ])

        # Quadra em metros → imagem em pixels
        self.matrix = cv2.getPerspectiveTransform(
            world_points,
            image_points,
        )

        # Imagem em pixels → quadra em metros
        self.inverse_matrix = cv2.getPerspectiveTransform(
            image_points,
            world_points,
        )

        return self.matrix

    def transform_point(self, point):
        """
        Converte um ponto da quadra, em metros,
        para um ponto da imagem, em pixels.
        """

        if self.matrix is None:
            raise ValueError(
                "A homografia ainda não foi calculada."
            )

        world_point = np.array(
            [[[point[0], point[1]]]],
            dtype=np.float32,
        )

        image_point = cv2.perspectiveTransform(
            world_point,
            self.matrix,
        )

        x = int(image_point[0][0][0])
        y = int(image_point[0][0][1])

        return (x, y)

    def transform(self, frame):
        #Cria uma Bird's-Eye View da quadra. A homografia corrige o plano do chão. Objetos verticais, como jogadores, podem parecer deformados na imagem transformada.

        pixels_per_meter = 40
        margin = 80

        court_width = int(
            self.court_model.width * pixels_per_meter
        )

        court_height = int(
            self.court_model.length * pixels_per_meter
        )

        output_width = court_width + (margin * 2)
        output_height = court_height + (margin * 2)

        source_points = np.float32([
            self.image_points.top_left,
            self.image_points.top_right,
            self.image_points.bottom_right,
            self.image_points.bottom_left,
        ])

        destination_points = np.float32([
            [margin, margin],
            [margin + court_width, margin],
            [
                margin + court_width,
                margin + court_height,
            ],
            [margin, margin + court_height],
        ])

        bird_eye_matrix = cv2.getPerspectiveTransform(
            source_points,
            destination_points,
        )

        bird_view = cv2.warpPerspective(
            frame,
            bird_eye_matrix,
            (output_width, output_height),
        )

        return bird_view

    def transform_point_to_world(self, point):
        # Converte um ponto da imagem (pixels) para a posição real na quadra (metros).
       
        if self.inverse_matrix is None:
            raise ValueError("A homografia ainda não foi calculada.")

        image_point = np.array(
            [[[point[0], point[1]]]],
            dtype=np.float32,
        )

        world_point = cv2.perspectiveTransform(
            image_point,
            self.inverse_matrix,
        )

        x = float(world_point[0][0][0])
        y = float(world_point[0][0][1])

        return (x, y)
    