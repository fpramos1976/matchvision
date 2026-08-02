import collections
import cv2
import numpy as np


class BallDetector:
    """
    Detector de bola de ténis robusto com:
    1. Filtro HSV para bola amarela.
    2. Motion Mask (diferença de frames).
    3. Blacklist de pontos estáticos (elimina linhas brancas e bolas no solo).
    4. Rastreio e suavização via Filtro de Kalman.
    """

    def __init__(self, max_buffer=30):
        self.max_buffer = max_buffer

        self.trajectory_pixels = collections.deque(maxlen=max_buffer)
        self.trajectory_world = collections.deque(maxlen=max_buffer)

        self.previous_gray = None

        # --- ESTADOS E CONTADORES ---
        self.kalman_initialized = False
        self.missed_frames = 0
        self.static_frames_count = 0
        self.last_confirmed_center = None
        self.frame_number = 0
        self.show_debug = True

        # Posições bloqueadas por serem estáticas (linhas, brilhos fixos, etc.)
        self.blocked_static_points = []

        # ---------------------------------------------------------
        # Filtro de Kalman
        # ---------------------------------------------------------
        self.kalman = cv2.KalmanFilter(4, 2)

        self.kalman.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32
        )

        self.kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )

        self.kalman.processNoiseCov = (
            np.array(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 4, 0],
                    [0, 0, 0, 4],
                ],
                dtype=np.float32,
            )
            * 0.03
        )

        self.kalman.measurementNoiseCov = (
            np.eye(2, dtype=np.float32) * 4.0
        )

        self.kalman.errorCovPost = np.eye(4, dtype=np.float32) * 20.0

    def reset_kalman(self):
        """Reinicializa o Kalman quando a bola fica perdida ou estática."""
        self.kalman_initialized = False
        self.missed_frames = 0
        self.static_frames_count = 0
        self.last_confirmed_center = None

        self.kalman.statePre = np.zeros((4, 1), dtype=np.float32)
        self.kalman.statePost = np.zeros((4, 1), dtype=np.float32)

    def _create_player_mask(self, frame_shape, player_boxes):
        """Cria uma máscara que remove as regiões dos jogadores."""
        height, width = frame_shape[:2]
        player_mask = np.full((height, width), 255, dtype=np.uint8)

        if not player_boxes:
            return player_mask

        for box in player_boxes:
            try:
                x1, y1, x2, y2 = map(int, box[:4])
                pad_x = 15
                pad_top = 10
                pad_bottom = 20

                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_top)
                x2 = min(width, x2 + pad_x)
                y2 = min(height, y2 + pad_bottom)

                player_mask[y1:y2, x1:x2] = 0
            except (TypeError, ValueError, IndexError):
                continue

        return player_mask

    def _create_yellow_mask(self, frame):
        #Procura a faixa amarela/esverdeada brilhante da bola, ignorando folhagens
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Exige mais Saturação (mínimo 60 em vez de 40) para ignorar árvores e sombras
        lower_yellow = np.array([25, 60, 90], dtype=np.uint8)
        upper_yellow = np.array([55, 255, 255], dtype=np.uint8)

        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        yellow_mask = cv2.dilate(yellow_mask, kernel, iterations=1)

        return yellow_mask

    def _create_motion_mask(self, gray):
        """Cria uma máscara pela diferença de movimento entre frames."""
        if self.previous_gray is None:
            self.previous_gray = gray.copy()
            return np.zeros_like(gray, dtype=np.uint8)

        difference = cv2.absdiff(gray, self.previous_gray)
        self.previous_gray = gray.copy()

        _, motion_mask = cv2.threshold(difference, 10, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        motion_mask = cv2.dilate(motion_mask, kernel, iterations=1)

        return motion_mask

    def _create_search_mask(self, image_shape, predicted_point):
        """Cria a janela de busca baseada no Kalman."""
        height, width = image_shape[:2]
        search_mask = np.full((height, width), 255, dtype=np.uint8)

        if predicted_point is None:
            return search_mask

        predicted_x, predicted_y = predicted_point

        if not (0 <= predicted_x < width and 0 <= predicted_y < height):
            return search_mask

        radius = min(300, 100 + (self.missed_frames * 30))
        search_mask[:] = 0
        cv2.circle(search_mask, predicted_point, radius, 255, -1)

        return search_mask

    def _is_inside_court(self, center, homography, court_model):
        """Verifica se o candidato está dentro dos limites da quadra."""
        if homography is None or court_model is None:
            return True, None

        try:
            world_x, world_y = homography.transform_point_to_world(center)
        except Exception:
            return False, None

        margin_x = 4.0
        margin_y = 4.0

        inside = (
            -margin_x <= world_x <= court_model.width + margin_x
            and -margin_y <= world_y <= court_model.length + margin_y
        )

        return inside, (world_x, world_y)

    def _is_point_blacklisted(self, center):
        """Verifica se o ponto está na lista de bloqueio de ruídos estáticos."""
        for blocked_pt in self.blocked_static_points:
            dist = np.hypot(center[0] - blocked_pt[0], center[1] - blocked_pt[1])
            if dist < 25.0:  # Raio de bloqueio ao redor da linha/ruído
                return True
        return False

    def _score_candidate(
        self,
        center,
        radius,
        area,
        circularity,
        predicted_point,
        yellow_value,
        motion_value,
    ):
        """Calcula a pontuação do candidato."""
        score = 0.0

        # --- Geometria ---
        ideal_radius = 3.5
        radius_difference = abs(radius - ideal_radius)
        score += max(0.0, 25.0 - (radius_difference * 4.0))

        ideal_area = 15.0
        area_difference = abs(area - ideal_area)
        score += max(0.0, 15.0 - (area_difference * 0.15))

        score += min(15.0, circularity * 20.0)

        # --- Cor Amarela (Obrigatória) ---
        if yellow_value > 0:
            score += 30.0
        else:
            score -= 40.0  # Desqualifica se não tiver tom amarelo

        # --- MOVIMENTO REAL ---
        if motion_value > 0:
            score += 40.0
        else:
            score -= 50.0  # Desqualifica se estiver parado (como as linhas da quadra)

        # --- Distância do Kalman ---
        if predicted_point is not None:
            distance = float(
                np.hypot(
                    center[0] - predicted_point[0],
                    center[1] - predicted_point[1],
                )
            )
            score += max(0.0, 30.0 - (distance * 0.15))

        return score

    def detect(
        self, frame, homography=None, court_model=None, player_boxes=None
    ):
        """Deteta a bola no frame atual."""
        self.frame_number += 1
        height, width = frame.shape[:2]

        # Limpa periodicamente a lista de pontos bloqueados antigos
        if self.frame_number % 120 == 0:
            self.blocked_static_points.clear()

        # 1. Previsão Kalman
        predicted_point = None
        if self.kalman_initialized:
            prediction = self.kalman.predict()
            predicted_x = int(prediction[0][0])
            predicted_y = int(prediction[1][0])

            if 0 <= predicted_x < width and 0 <= predicted_y < height:
                predicted_point = (predicted_x, predicted_y)

        # 2. Cinza
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # 3. Máscaras
        yellow_mask = self._create_yellow_mask(frame)
        motion_mask = self._create_motion_mask(gray)

        # --- MÁSCARA DE REGIÃO DE INTERESSE (ROI) ---
        # Corta o topo (árvores + lona/bancadas do fundo) a partir de ~30% da altura
        roi_mask = np.zeros((height, width), dtype=np.uint8)
        min_y_roi = int(height * 0.30)  # <--- Ajustado de 0.18 para 0.30
        roi_mask[min_y_roi:, :] = 255

        player_mask = self._create_player_mask(frame.shape, player_boxes)
        search_mask = self._create_search_mask(frame.shape, predicted_point)

        # 4. Combinação Inteligente
        # Se o Kalman já está a rastrear a bola, usamos a janela de busca (search_mask).
        # Se NÃO está a rastrear, exigimos movimento (motion_mask) para não pegar placas/lonas fixas!
        if self.kalman_initialized:
            combined_mask = cv2.bitwise_and(yellow_mask, search_mask)
        else:
            # Exige que a cor amarela esteja em movimento para iniciar o rastreio
            yellow_in_motion = cv2.bitwise_and(yellow_mask, motion_mask)
            combined_mask = cv2.bitwise_and(yellow_mask, yellow_in_motion)

        combined_mask = cv2.bitwise_and(combined_mask, player_mask)
        combined_mask = cv2.bitwise_and(combined_mask, roi_mask)
        # 5. Contornos
        contours, _ = cv2.findContours(
            combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 0.5 or area > 200.0:
                continue

            (x, y), radius = cv2.minEnclosingCircle(contour)
            if radius < 0.7 or radius > 10.0:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue

            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.05:
                continue

            center = (int(x), int(y))

            # Ignora pontos bloqueados na Blacklist (ex: pedaço de linha estático)
            if self._is_point_blacklisted(center):
                continue

            inside_court, candidate_world = self._is_inside_court(
                center, homography, court_model
            )
            if not inside_court:
                continue

            center_x = min(width - 1, max(0, center[0]))
            center_y = min(height - 1, max(0, center[1]))

            yellow_value = int(yellow_mask[center_y, center_x])
            motion_value = int(motion_mask[center_y, center_x])

            score = self._score_candidate(
                center,
                radius,
                area,
                circularity,
                predicted_point,
                yellow_value,
                motion_value,
            )

            candidates.append(
                {
                    "center": center,
                    "radius": radius,
                    "area": area,
                    "circularity": circularity,
                    "score": score,
                    "motion_value": motion_value,
                    "world_pos": candidate_world,
                }
            )

        # 6. Seleciona o melhor candidato
        ball_center = None
        world_pos = None

        if candidates:
            candidates.sort(key=lambda item: item["score"], reverse=True)

            for candidate in candidates:
                cand_center = candidate["center"]

                # Validação de deslocamento (Anti-Jump)
                if (
                    self.last_confirmed_center is not None
                    and self.missed_frames < 3
                ):
                    dist_from_last = np.hypot(
                        cand_center[0] - self.last_confirmed_center[0],
                        cand_center[1] - self.last_confirmed_center[1],
                    )

                    max_allowed_jump = 55.0
                    if dist_from_last > max_allowed_jump:
                        continue

                # Para iniciar o rastreio do zero, EXIGE movimento real!
                if not self.kalman_initialized and candidate["motion_value"] == 0:
                    continue

                if candidate["score"] >= 25.0:
                    ball_center = cand_center
                    world_pos = candidate["world_pos"]
                    break

        # 7. Atualiza Kalman e verifica se está congelado na linha
        if ball_center is not None:
            if self.last_confirmed_center is not None:
                dist_stuck = np.hypot(
                    ball_center[0] - self.last_confirmed_center[0],
                    ball_center[1] - self.last_confirmed_center[1],
                )
                if dist_stuck < 2.0:
                    self.static_frames_count += 1
                else:
                    self.static_frames_count = 0

            # Se ficou preso no mesmo local por mais de 3 frames: BLOQUEIA o ponto e RESETA!
            if self.static_frames_count >= 3:
                self.blocked_static_points.append(ball_center)
                self.reset_kalman()
                self.trajectory_pixels.clear()
                self.trajectory_world.clear()
                return None, None

            measurement = np.array(
                [
                    [np.float32(ball_center[0])],
                    [np.float32(ball_center[1])],
                ],
                dtype=np.float32,
            )

            if not self.kalman_initialized:
                initial_state = np.array(
                    [
                        [np.float32(ball_center[0])],
                        [np.float32(ball_center[1])],
                        [0.0],
                        [0.0],
                    ],
                    dtype=np.float32,
                )
                self.kalman.statePre = initial_state.copy()
                self.kalman.statePost = initial_state.copy()
                self.kalman_initialized = True

            self.kalman.correct(measurement)
            self.missed_frames = 0
            self.last_confirmed_center = ball_center

            self.trajectory_pixels.appendleft(ball_center)
            self.trajectory_world.appendleft(world_pos)
        else:
            self.static_frames_count = 0
            self.missed_frames += 1
            self.trajectory_pixels.appendleft(None)
            self.trajectory_world.appendleft(None)

            if self.missed_frames > 15:
                self.reset_kalman()
                self.trajectory_pixels.clear()
                self.trajectory_world.clear()

        # 8. Visualização Debug
        if self.show_debug:
            debug_candidates = frame.copy()

            for candidate in candidates:
                cnt_center = candidate["center"]
                radius = int(max(2, candidate["radius"]))
                cv2.circle(
                    debug_candidates,
                    cnt_center,
                    radius + 3,
                    (0, 255, 255),
                    1,
                )
                cv2.putText(
                    debug_candidates,
                    f'{candidate["score"]:.0f}',
                    (cnt_center[0] + 5, cnt_center[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            # Desenha os pontos bloqueados (linhas estáticas ignoradas)
            for pt in self.blocked_static_points:
                cv2.circle(debug_candidates, pt, 12, (0, 0, 0), 2)

            if predicted_point is not None:
                cv2.circle(
                    debug_candidates, predicted_point, 9, (255, 0, 255), 2
                )

            if ball_center is not None:
                cv2.circle(debug_candidates, ball_center, 8, (0, 0, 255), 2)

            cv2.imshow("Ball Debug - Yellow", yellow_mask)
            cv2.imshow("Ball Debug - Motion", motion_mask)
            cv2.imshow("Ball Debug - Combined", combined_mask)
            cv2.imshow("Ball Debug - Candidates", debug_candidates)

        return ball_center, world_pos

    def draw_ball_trail(self, frame):
        """Desenha a trajetória apenas entre pontos válidos."""
        for index in range(1, len(self.trajectory_pixels)):
            point_1 = self.trajectory_pixels[index - 1]
            point_2 = self.trajectory_pixels[index]

            if point_1 is not None and point_2 is not None:
                cv2.line(
                    frame, point_1, point_2, (0, 255, 255), 2, cv2.LINE_AA
                )

        if self.trajectory_pixels and self.trajectory_pixels[0] is not None:
            center = self.trajectory_pixels[0]
            cv2.circle(frame, center, 6, (0, 0, 255), 2)
            cv2.circle(frame, center, 2, (0, 255, 255), -1)

        return frame