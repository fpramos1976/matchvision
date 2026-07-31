import collections
import cv2
import numpy as np


class BallDetector:

    def __init__(self, max_buffer=20):
        """Detector de Bola de Tênis Otimizado com Filtro de Kalman.

        Impede desvios abruptos e rejeita falsos positivos no saibro.
        """
        self.max_buffer = max_buffer
        self.trajectory_pixels = collections.deque(maxlen=max_buffer)
        self.trajectory_world = collections.deque(maxlen=max_buffer)

        # Subtrator de Fundo para regiões em movimento
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=30, varThreshold=16, detectShadows=False
        )

        # -----------------------------------------------------------------
        # Inicialização do Filtro de Kalman (Estado: [x, y, dx, dy])
        # -----------------------------------------------------------------
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], np.float32
        )
        self.kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )
        # Ruído de processo/medição ajustado para dinâmicas de tênis
        self.kalman.processNoiseCov = (
            np.array(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 5, 0],
                    [0, 0, 0, 5],
                ],
                np.float32,
            )
            * 0.03
        )

        self.kalman_initialized = False
        self.missed_frames = 0

    def reset_kalman(self):
        """Reseta o rastreador de Kalman quando a bola é perdida por muito

        tempo.
        """
        self.kalman_initialized = False
        self.missed_frames = 0

    def detect(
        self, frame, homography=None, court_model=None, player_boxes=None
    ):
        h, w = frame.shape[:2]

        # 1. Predição de Kalman para o frame atual
        predicted_pt = None
        if self.kalman_initialized:
            prediction = self.kalman.predict()
            predicted_pt = (int(prediction[0][0]), int(prediction[1][0]))

        # 2. Subtração de fundo
        fg_mask = self.bg_subtractor.apply(frame)

        # Morph para limpar pequenos ruídos soltos
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        # 3. MÁSCARA EXPANDIDA DOS JOGADORES (Elimina tênis, pernas e sombras dos pés)
        if player_boxes:
            for box in player_boxes:
                try:
                    x1, y1, x2, y2 = map(int, box[:4])

                    # Damos uma margem extra grande (padding) principalmente para BAIXO (+40px)
                    # para cobrir o movimento dos sapatos e o saibro levantado no chão
                    pad_x = 30
                    pad_y_top = 25
                    pad_y_bottom = 45

                    x1_p, y1_p = max(0, x1 - pad_x), max(0, y1 - pad_y_top)
                    x2_p, y2_p = min(w, x2 + pad_x), min(h, y2 + pad_y_bottom)

                    fg_mask[y1_p:y2_p, x1_p:x2_p] = 0
                except Exception:
                    continue

        # 4. Filtro de Brilho em HSV (Busca o amarelo/branco vivo da bola de tênis)
        # Exige cor/brilho real para ignorar sombras e marcas escuras no saibro
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, bright_mask = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
        combined_mask = cv2.bitwise_and(fg_mask, bright_mask)

        # 5. Se já temos predição, busca SOMENTE na janela do Kalman
        if self.kalman_initialized and predicted_pt:
            search_mask = np.zeros_like(combined_mask)
            cv2.circle(
                search_mask,
                predicted_pt,
                max(45, 18 * (self.missed_frames + 1)),
                255,
                -1,
            )
            combined_mask = cv2.bitwise_and(combined_mask, search_mask)

        # 6. Busca por contornos
        cnts, _ = cv2.findContours(
            combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []
        if cnts:
            for c in cnts:
                area = cv2.contourArea(c)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                # Formato bem restrito para bolinha
                if 0.8 <= radius <= 7.0 and 1 <= area <= 110:
                    candidates.append(((int(x), int(y)), radius, area))

        ball_center = None
        world_pos = None

        if candidates:
            if self.kalman_initialized and predicted_pt:
                candidates.sort(
                    key=lambda item: (item[0][0] - predicted_pt[0]) ** 2
                    + (item[0][1] - predicted_pt[1]) ** 2
                )
            elif len(self.trajectory_pixels) > 0 and self.trajectory_pixels[0]:
                last = self.trajectory_pixels[0]
                candidates.sort(
                    key=lambda item: (item[0][0] - last[0]) ** 2
                    + (item[0][1] - last[1]) ** 2
                )

            for center, radius, area in candidates:
                cand_world = None
                if homography and court_model:
                    cand_world = homography.transform_point_to_world(center)
                    wx, wy = cand_world
                    if not (
                        -4.0 <= wx <= court_model.width + 4.0
                        and -3.0 <= wy <= court_model.length + 3.0
                    ):
                        continue

                ball_center = center
                world_pos = cand_world
                break

        # 7. Atualização do Filtro de Kalman e Histórico
        if ball_center:
            measurement = np.array(
                [[np.float32(ball_center[0])], [np.float32(ball_center[1])]]
            )

            if not self.kalman_initialized:
                self.kalman.statePre = np.array(
                    [
                        [np.float32(ball_center[0])],
                        [np.float32(ball_center[1])],
                        [0],
                        [0],
                    ],
                    np.float32,
                )
                self.kalman.statePost = self.kalman.statePre.copy()
                self.kalman_initialized = True

            self.kalman.correct(measurement)
            self.trajectory_pixels.appendleft(ball_center)
            if world_pos:
                self.trajectory_world.appendleft(world_pos)

            self.missed_frames = 0
        else:
            self.missed_frames += 1

            if self.kalman_initialized and self.missed_frames <= 4:
                self.trajectory_pixels.appendleft(predicted_pt)
                if homography and predicted_pt:
                    self.trajectory_world.appendleft(
                        homography.transform_point_to_world(predicted_pt)
                    )
            else:
                self.trajectory_pixels.appendleft(None)
                self.trajectory_world.appendleft(None)

            if self.missed_frames > 8:
                self.reset_kalman()
                self.trajectory_pixels.clear()
                self.trajectory_world.clear()

        return ball_center, world_pos

    def draw_ball_trail(self, frame):
        """Desenha a linha suave calculada pelo Kalman sem os saltos no

        saibro.
        """
        for i in range(1, len(self.trajectory_pixels)):
            p1 = self.trajectory_pixels[i - 1]
            p2 = self.trajectory_pixels[i]

            if p1 is not None and p2 is not None:
                cv2.line(frame, p1, p2, (0, 255, 255), 2)

        if self.trajectory_pixels and self.trajectory_pixels[0]:
            center = self.trajectory_pixels[0]
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            cv2.circle(frame, center, 2, (0, 255, 255), -1)

        return frame