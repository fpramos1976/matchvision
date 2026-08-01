import collections

import cv2
import numpy as np


class BallDetector:
    """
    Detector de bola de ténis baseado em:

    1. Cor amarela em HSV.
    2. Diferença entre frames.
    3. Exclusão das regiões dos jogadores.
    4. Tamanho e formato dos candidatos.
    5. Previsão com filtro de Kalman.
    6. Pontuação combinada dos candidatos.

    A deteção não exige obrigatoriamente que a bola seja amarela E
    esteja em movimento. Isso é importante em vídeos com FPS elevado,
    onde a diferença entre frames pode ser muito pequena.
    """

    def __init__(self, max_buffer=30):
        self.max_buffer = max_buffer

        # Histórico apenas das posições realmente detetadas.
        self.trajectory_pixels = collections.deque(
            maxlen=max_buffer
        )

        self.trajectory_world = collections.deque(
            maxlen=max_buffer
        )

        # Frame anterior em escala de cinza.
        self.previous_gray = None

        # ---------------------------------------------------------
        # Filtro de Kalman
        #
        # Estado:
        # [x, y, velocidade_x, velocidade_y]
        #
        # Medição:
        # [x, y]
        # ---------------------------------------------------------
        self.kalman = cv2.KalmanFilter(
            4,
            2,
        )

        self.kalman.measurementMatrix = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ],
            dtype=np.float32,
        )

        self.kalman.transitionMatrix = np.array(
            [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

        # Permite mudanças rápidas de direção e velocidade.
        self.kalman.processNoiseCov = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 12, 0],
                [0, 0, 0, 12],
            ],
            dtype=np.float32,
        ) * 0.05

        self.kalman.measurementNoiseCov = (
            np.eye(
                2,
                dtype=np.float32,
            )
            * 4.0
        )

        self.kalman.errorCovPost = (
            np.eye(
                4,
                dtype=np.float32,
            )
            * 20.0
        )

        self.kalman_initialized = False
        self.missed_frames = 0

        self.last_confirmed_center = None

        self.frame_number = 0

        # Janelas para analisar as máscaras.
        self.show_debug = True

    def reset_kalman(self):
        """Reinicializa o Kalman quando a bola fica perdida."""

        self.kalman_initialized = False
        self.missed_frames = 0
        self.last_confirmed_center = None

        self.kalman.statePre = np.zeros(
            (4, 1),
            dtype=np.float32,
        )

        self.kalman.statePost = np.zeros(
            (4, 1),
            dtype=np.float32,
        )

    def _create_player_mask(
        self,
        frame_shape,
        player_boxes,
    ):
        """
        Cria uma máscara que remove as regiões dos jogadores.

        A margem é pequena para não esconder a bola quando ela passa
        perto do corpo do jogador.
        """

        height, width = frame_shape[:2]

        player_mask = np.full(
            (height, width),
            255,
            dtype=np.uint8,
        )

        if not player_boxes:
            return player_mask

        for box in player_boxes:
            try:
                x1, y1, x2, y2 = map(
                    int,
                    box[:4],
                )

                pad_x = 12
                pad_top = 8
                pad_bottom = 15

                x1 = max(
                    0,
                    x1 - pad_x,
                )

                y1 = max(
                    0,
                    y1 - pad_top,
                )

                x2 = min(
                    width,
                    x2 + pad_x,
                )

                y2 = min(
                    height,
                    y2 + pad_bottom,
                )

                player_mask[
                    y1:y2,
                    x1:x2,
                ] = 0

            except (
                TypeError,
                ValueError,
                IndexError,
            ):
                continue

        return player_mask

    def _create_yellow_mask(
        self,
        frame,
    ):
        """
        Procura a faixa amarela da bola.

        A faixa é mais ampla do que a versão anterior porque a bola
        pode perder saturação devido ao blur, sombra e compressão.
        """

        hsv = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2HSV,
        )

        lower_yellow = np.array(
            [15, 35, 70],
            dtype=np.uint8,
        )

        upper_yellow = np.array(
            [50, 255, 255],
            dtype=np.uint8,
        )

        yellow_mask = cv2.inRange(
            hsv,
            lower_yellow,
            upper_yellow,
        )

        # Não usamos MORPH_OPEN.
        #
        # Uma bola distante pode ter apenas 1 ou 2 pixels e o OPEN
        # pode apagá-la completamente.
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3),
        )

        yellow_mask = cv2.dilate(
            yellow_mask,
            kernel,
            iterations=1,
        )

        return yellow_mask

    def _create_bright_mask(
        self,
        frame,
    ):
        """
        Cria uma máscara para pontos claros.

        Ajuda quando a bola perde a cor amarela e aparece quase branca
        por causa da luz ou da compressão.
        """

        hsv = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2HSV,
        )

        lower_bright = np.array(
            [0, 0, 125],
            dtype=np.uint8,
        )

        upper_bright = np.array(
            [179, 130, 255],
            dtype=np.uint8,
        )

        return cv2.inRange(
            hsv,
            lower_bright,
            upper_bright,
        )

    def _create_motion_mask(
        self,
        gray,
    ):
        """
        Cria uma máscara pela diferença entre frames.

        O limiar foi reduzido porque o vídeo tem aproximadamente
        120 FPS e a bola pode mudar muito pouco entre dois frames.
        """

        if self.previous_gray is None:
            self.previous_gray = gray.copy()

            return np.zeros_like(
                gray,
                dtype=np.uint8,
            )

        difference = cv2.absdiff(
            gray,
            self.previous_gray,
        )

        self.previous_gray = gray.copy()

        _, motion_mask = cv2.threshold(
            difference,
            8,
            255,
            cv2.THRESH_BINARY,
        )

        # Não usamos MORPH_OPEN.
        #
        # Mantemos os pequenos pontos que podem ser a bola.
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3),
        )

        motion_mask = cv2.dilate(
            motion_mask,
            kernel,
            iterations=1,
        )

        return motion_mask

    def _create_search_mask(
        self,
        image_shape,
        predicted_point,
    ):
        """
        Cria a área de procura do Kalman.

        Se a previsão estiver fora da imagem, a procura é feita em
        toda a imagem.
        """

        height, width = image_shape[:2]

        search_mask = np.full(
            (height, width),
            255,
            dtype=np.uint8,
        )

        if predicted_point is None:
            return search_mask

        predicted_x, predicted_y = predicted_point

        if not (
            0 <= predicted_x < width
            and
            0 <= predicted_y < height
        ):
            return search_mask

        radius = min(
            350,
            110 + (
                self.missed_frames * 35
            ),
        )

        search_mask[:] = 0

        cv2.circle(
            search_mask,
            predicted_point,
            radius,
            255,
            -1,
        )

        return search_mask

    def _is_inside_court(
        self,
        center,
        homography,
        court_model,
    ):
        """
        Verifica se o candidato está dentro da quadra ou numa margem
        externa.
        """

        if (
            homography is None
            or court_model is None
        ):
            return True, None

        try:
            world_x, world_y = (
                homography.transform_point_to_world(
                    center
                )
            )

        except Exception:
            return False, None

        margin_x = 5.0
        margin_y = 5.0

        inside = (
            -margin_x
            <= world_x
            <= court_model.width + margin_x
            and
            -margin_y
            <= world_y
            <= court_model.length + margin_y
        )

        return inside, (
            world_x,
            world_y,
        )

    def _score_candidate(
        self,
        center,
        radius,
        area,
        circularity,
        predicted_point,
        yellow_value,
        motion_value,
        bright_value,
    ):
        """
        Calcula a pontuação de um candidato.

        Cor, movimento, tamanho, forma e distância da previsão são
        avaliados separadamente.
        """

        score = 0.0

        # ---------------------------------------------------------
        # Tamanho
        # ---------------------------------------------------------
        ideal_radius = 3.0

        radius_difference = abs(
            radius - ideal_radius
        )

        score += max(
            0.0,
            25.0 - (
                radius_difference * 4.0
            ),
        )

        # ---------------------------------------------------------
        # Área
        # ---------------------------------------------------------
        ideal_area = 15.0

        area_difference = abs(
            area - ideal_area
        )

        score += max(
            0.0,
            15.0 - (
                area_difference * 0.15
            ),
        )

        # ---------------------------------------------------------
        # Forma
        # ---------------------------------------------------------
        score += min(
            15.0,
            circularity * 20.0,
        )

        # ---------------------------------------------------------
        # Cor
        # ---------------------------------------------------------
        if yellow_value > 0:
            score += 30.0

        # ---------------------------------------------------------
        # Movimento
        # ---------------------------------------------------------
        if motion_value > 0:
            score += 25.0

        # ---------------------------------------------------------
        # Brilho
        # ---------------------------------------------------------
        if bright_value > 0:
            score += 8.0

        # ---------------------------------------------------------
        # Previsão do Kalman
        # ---------------------------------------------------------
        if predicted_point is not None:
            distance = float(
                np.hypot(
                    center[0]
                    - predicted_point[0],
                    center[1]
                    - predicted_point[1],
                )
            )

            score += max(
                0.0,
                35.0 - (
                    distance * 0.15
                ),
            )

        return score

    def detect(
        self,
        frame,
        homography=None,
        court_model=None,
        player_boxes=None,
    ):
        """
        Deteta a bola no frame atual.

        Retorna:

        ball_center:
            posição da bola em pixels ou None.

        world_pos:
            posição projetada na quadra ou None.
        """

        self.frame_number += 1

        height, width = frame.shape[:2]

        # ---------------------------------------------------------
        # 1. Previsão do Kalman
        # ---------------------------------------------------------
        predicted_point = None

        if self.kalman_initialized:
            prediction = self.kalman.predict()

            predicted_x = int(
                prediction[0][0]
            )

            predicted_y = int(
                prediction[1][0]
            )

            if (
                0 <= predicted_x < width
                and
                0 <= predicted_y < height
            ):
                predicted_point = (
                    predicted_x,
                    predicted_y,
                )

        # ---------------------------------------------------------
        # 2. Frame em cinza
        # ---------------------------------------------------------
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        gray = cv2.GaussianBlur(
            gray,
            (3, 3),
            0,
        )

        # ---------------------------------------------------------
        # 3. Máscaras
        # ---------------------------------------------------------
        yellow_mask = self._create_yellow_mask(
            frame
        )

        bright_mask = self._create_bright_mask(
            frame
        )

        motion_mask = self._create_motion_mask(
            gray
        )

        player_mask = self._create_player_mask(
            frame.shape,
            player_boxes,
        )

        search_mask = self._create_search_mask(
            frame.shape,
            predicted_point,
        )

        # ---------------------------------------------------------
        # 4. Combinação das máscaras
        # ---------------------------------------------------------
        #
        # Aceitamos:
        #
        # A) amarelo + movimento
        # B) amarelo sozinho
        # C) movimento + brilho
        #
        # A bola não precisa passar por todos os filtros.
        yellow_motion = cv2.bitwise_and(
            yellow_mask,
            motion_mask,
        )

        bright_motion = cv2.bitwise_and(
            bright_mask,
            motion_mask,
        )

        combined_mask = cv2.bitwise_or(
            yellow_motion,
            yellow_mask,
        )

        combined_mask = cv2.bitwise_or(
            combined_mask,
            bright_motion,
        )

        # Remove regiões dos jogadores.
        combined_mask = cv2.bitwise_and(
            combined_mask,
            player_mask,
        )

        # Aplica a região prevista pelo Kalman.
        combined_mask = cv2.bitwise_and(
            combined_mask,
            search_mask,
        )

        # ---------------------------------------------------------
        # 5. Procura por contornos
        # ---------------------------------------------------------
        contours, _ = cv2.findContours(
            combined_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        candidates = []

        for contour in contours:
            area = cv2.contourArea(
                contour
            )

            if (
                area < 0.5
                or area > 250.0
            ):
                continue

            (
                (x, y),
                radius,
            ) = cv2.minEnclosingCircle(
                contour
            )

            if (
                radius < 0.7
                or radius > 12.0
            ):
                continue

            perimeter = cv2.arcLength(
                contour,
                True,
            )

            if perimeter <= 0:
                continue

            circularity = (
                4.0
                * np.pi
                * area
                / (
                    perimeter
                    * perimeter
                )
            )

            # A bola pode ficar muito deformada.
            if circularity < 0.04:
                continue

            center = (
                int(x),
                int(y),
            )

            inside_court, candidate_world = (
                self._is_inside_court(
                    center,
                    homography,
                    court_model,
                )
            )

            if not inside_court:
                continue

            center_x = min(
                width - 1,
                max(
                    0,
                    center[0],
                ),
            )

            center_y = min(
                height - 1,
                max(
                    0,
                    center[1],
                ),
            )

            yellow_value = int(
                yellow_mask[
                    center_y,
                    center_x,
                ]
            )

            motion_value = int(
                motion_mask[
                    center_y,
                    center_x,
                ]
            )

            bright_value = int(
                bright_mask[
                    center_y,
                    center_x,
                ]
            )

            score = self._score_candidate(
                center,
                radius,
                area,
                circularity,
                predicted_point,
                yellow_value,
                motion_value,
                bright_value,
            )

            candidates.append(
                {
                    "center": center,
                    "radius": radius,
                    "area": area,
                    "circularity": circularity,
                    "score": score,
                    "world_pos": candidate_world,
                }
            )

        # ---------------------------------------------------------
        # 6. Seleciona o melhor candidato
        # ---------------------------------------------------------
        ball_center = None
        world_pos = None

        if candidates:
            candidates.sort(
                key=lambda item: (
                    item["score"]
                ),
                reverse=True,
            )

            best_candidate = candidates[0]

            minimum_score = 28.0

            if (
                best_candidate["score"]
                >= minimum_score
            ):
                ball_center = (
                    best_candidate["center"]
                )

                world_pos = (
                    best_candidate[
                        "world_pos"
                    ]
                )

        # ---------------------------------------------------------
        # 7. Atualiza o Kalman
        # ---------------------------------------------------------
        if ball_center is not None:
            measurement = np.array(
                [
                    [
                        np.float32(
                            ball_center[0]
                        )
                    ],
                    [
                        np.float32(
                            ball_center[1]
                        )
                    ],
                ],
                dtype=np.float32,
            )

            if not self.kalman_initialized:
                initial_state = np.array(
                    [
                        [
                            np.float32(
                                ball_center[0]
                            )
                        ],
                        [
                            np.float32(
                                ball_center[1]
                            )
                        ],
                        [0.0],
                        [0.0],
                    ],
                    dtype=np.float32,
                )

                self.kalman.statePre = (
                    initial_state.copy()
                )

                self.kalman.statePost = (
                    initial_state.copy()
                )

                self.kalman_initialized = True

            self.kalman.correct(
                measurement
            )

            self.missed_frames = 0

            self.last_confirmed_center = (
                ball_center
            )

            self.trajectory_pixels.appendleft(
                ball_center
            )

            self.trajectory_world.appendleft(
                world_pos
            )

        else:
            self.missed_frames += 1

            self.trajectory_pixels.appendleft(
                None
            )

            self.trajectory_world.appendleft(
                None
            )

            if (
                self.missed_frames > 20
            ):
                self.reset_kalman()

                self.trajectory_pixels.clear()

                self.trajectory_world.clear()

        # ---------------------------------------------------------
        # 8. Janelas de debug
        # ---------------------------------------------------------
        if self.show_debug:
            debug_candidates = frame.copy()

            for candidate in candidates:
                center = candidate[
                    "center"
                ]

                radius = int(
                    max(
                        2,
                        candidate[
                            "radius"
                        ],
                    )
                )

                cv2.circle(
                    debug_candidates,
                    center,
                    radius + 3,
                    (0, 255, 255),
                    1,
                )

                cv2.putText(
                    debug_candidates,
                    (
                        f'{candidate["score"]:.0f}'
                    ),
                    (
                        center[0] + 5,
                        center[1] - 5,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            if predicted_point is not None:
                cv2.circle(
                    debug_candidates,
                    predicted_point,
                    9,
                    (255, 0, 255),
                    2,
                )

            if ball_center is not None:
                cv2.circle(
                    debug_candidates,
                    ball_center,
                    8,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow(
                "Ball Debug - Yellow",
                yellow_mask,
            )

            cv2.imshow(
                "Ball Debug - Bright",
                bright_mask,
            )

            cv2.imshow(
                "Ball Debug - Motion",
                motion_mask,
            )

            cv2.imshow(
                "Ball Debug - Combined",
                combined_mask,
            )

            cv2.imshow(
                "Ball Debug - Candidates",
                debug_candidates,
            )

        return (
            ball_center,
            world_pos,
        )

    def draw_ball_trail(
        self,
        frame,
    ):
        """
        Desenha apenas as ligações entre deteções reais.

        Quando a bola desaparece, não é criada uma linha falsa.
        """

        for index in range(
            1,
            len(
                self.trajectory_pixels
            ),
        ):
            point_1 = (
                self.trajectory_pixels[
                    index - 1
                ]
            )

            point_2 = (
                self.trajectory_pixels[
                    index
                ]
            )

            if (
                point_1 is not None
                and point_2 is not None
            ):
                cv2.line(
                    frame,
                    point_1,
                    point_2,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        if (
            self.trajectory_pixels
            and
            self.trajectory_pixels[0]
            is not None
        ):
            center = (
                self.trajectory_pixels[0]
            )

            cv2.circle(
                frame,
                center,
                6,
                (0, 0, 255),
                2,
            )

            cv2.circle(
                frame,
                center,
                2,
                (0, 255, 255),
                -1,
            )

        return frame
    