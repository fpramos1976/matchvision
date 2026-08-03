import collections
import cv2
import numpy as np


class BallDetector:

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

        self.blocked_static_points = []
        self.bounces = []

        # ---------------------------------------------------------
        # Filtro de Kalman (4 estados: x, y, vx, vy | 2 medições: x, y)
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
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 8, 0], [0, 0, 0, 8]],
                dtype=np.float32,
            )
            * 0.05
        )
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 4.0
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32) * 20.0

    def reset_kalman(self):
        """Reinicializa o Kalman quando a bola fica perdida ou estática."""
        self.kalman_initialized = False
        self.missed_frames = 0
        self.static_frames_count = 0
        self.last_confirmed_center = None

        self.kalman.statePre = np.zeros((4, 1), dtype=np.float32)
        self.kalman.statePost = np.zeros((4, 1), dtype=np.float32)

    def check_bounce(self, court_model):
        """Analisa a variação de vetor para detetar ressaltos reais com menor taxa de falso positivo."""
        valid_points = [p for p in self.trajectory_world if p is not None]
        if len(valid_points) < 4:
            return None

        p1, p2, p3, p4 = (
            valid_points[0],
            valid_points[1],
            valid_points[2],
            valid_points[3],
        )

        # Variação temporal
        dy1 = p2[1] - p1[1]
        dy2 = p3[1] - p2[1]
        dy3 = p4[1] - p3[1]

        # Inversão nítida de sentido no eixo longitudinal
        if (dy1 * dy2 < 0) or (dy2 * dy3 < 0):
            bounce_point = p2

            if self.bounces:
                last_b_pos, _ = self.bounces[-1]
                dist = np.hypot(
                    bounce_point[0] - last_b_pos[0],
                    bounce_point[1] - last_b_pos[1],
                )
                if dist < 1.0:  # Evita duplicados a menos de 1 metro
                    return None

            is_in = (
                0.0 <= bounce_point[0] <= court_model.width
                and 0.0 <= bounce_point[1] <= court_model.length
            )

            bounce_info = (bounce_point, is_in)
            self.bounces.append(bounce_info)
            return bounce_info

        return None

    def _create_player_mask(self, frame_shape, player_boxes):
        """Oculta o jogador E a sua sombra projetada no chão (diagonal inferior esquerda)."""
        height, width = frame_shape[:2]
        player_mask = np.full((height, width), 255, dtype=np.uint8)

        if not player_boxes:
            return player_mask

        for box in player_boxes:
            try:
                if hasattr(box, "xyxy"):
                    coords = box.xyxy[0].cpu().numpy()
                elif isinstance(box, (list, tuple, np.ndarray)):
                    coords = box[:4]
                else:
                    continue

                x1, y1, x2, y2 = map(int, coords)

                # O sol projeta a sombra para a ESQUERDA e para BAIXO
                pad_x_left = 120  # Expande bastante para a esquerda (zona da sombra)
                pad_x_right = 40
                pad_top = 40
                pad_bottom = 60

                px1 = max(0, x1 - pad_x_left)
                py1 = max(0, y1 - pad_top)
                px2 = min(width, x2 + pad_x_right)
                py2 = min(height, y2 + pad_bottom)

                # Pinta a área do jogador + sombra a PRETO (0) para ignorar
                player_mask[py1:py2, px1:px2] = 0
            except Exception:
                continue

        return player_mask

    def _create_yellow_mask(self, frame):
        """Mascara HSV para capturar a bola amarela."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array([14, 30, 40], dtype=np.uint8)
        upper_yellow = np.array([65, 255, 255], dtype=np.uint8)

        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        yellow_mask = cv2.dilate(yellow_mask, kernel, iterations=1)

        return yellow_mask

    def _create_motion_mask(self, gray):
        """Máscara de movimento filtrando sombras escuras do chão."""
        if self.previous_gray is None:
            self.previous_gray = gray.copy()
            return np.zeros_like(gray, dtype=np.uint8)

        difference = cv2.absdiff(gray, self.previous_gray)

        # Threshold de movimento
        _, motion_mask = cv2.threshold(difference, 12, 255, cv2.THRESH_BINARY)

        # FILTRO DE SOMBRAS:
        # Pixels muito escuros (< 70) são sombras no chão -> elimina!
        _, bright_mask = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)
        motion_mask = cv2.bitwise_and(motion_mask, bright_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        motion_mask = cv2.dilate(motion_mask, kernel, iterations=1)

        self.previous_gray = gray.copy()
        return motion_mask

    def _create_search_mask(self, image_shape, predicted_point):
        """Cria uma região circular contida em redor da previsão do Kalman."""
        height, width = image_shape[:2]
        search_mask = np.full((height, width), 255, dtype=np.uint8)

        if predicted_point is None:
            return search_mask

        predicted_x, predicted_y = predicted_point
        if not (0 <= predicted_x < width and 0 <= predicted_y < height):
            return search_mask

        # Limitado a no máximo 120px para não apanhar metade da imagem quando a bola falha
        radius = min(120, 60 + (self.missed_frames * 10))
        search_mask[:] = 0
        cv2.circle(search_mask, predicted_point, radius, 255, -1)

        return search_mask

    def _is_inside_court(self, center, homography, court_model):
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
        for blocked_pt in self.blocked_static_points:
            dist = np.hypot(center[0] - blocked_pt[0], center[1] - blocked_pt[1])
            if dist < 25.0:
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
        score = 0.0

        ideal_radius = 3.5
        radius_difference = abs(radius - ideal_radius)
        score += max(0.0, 25.0 - (radius_difference * 4.0))

        ideal_area = 15.0
        area_difference = abs(area - ideal_area)
        score += max(0.0, 15.0 - (area_difference * 0.15))

        score += min(15.0, circularity * 20.0)

        if yellow_value > 0:
            score += 30.0
        else:
            score -= 30.0

        if motion_value > 0:
            score += 40.0
        else:
            score -= 40.0

        if predicted_point is not None:
            distance = float(
                np.hypot(
                    center[0] - predicted_point[0],
                    center[1] - predicted_point[1],
                )
            )
            score += max(0.0, 35.0 - (distance * 0.2))

        return score

    def detect(
        self, frame, homography=None, court_model=None, player_boxes=None
    ):
        self.frame_number += 1
        height, width = frame.shape[:2]

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

        # 2. Imagem em tons de cinza
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # 3. Gerar Máscaras
        yellow_mask = self._create_yellow_mask(frame)
        motion_mask = self._create_motion_mask(gray)

        # ROI MAIS RESTRITA: corta o topo da imagem (árvores/fundo)
        roi_mask = np.zeros((height, width), dtype=np.uint8)
        min_y_roi = int(height * 0.35)  # Corta o fundo ruidoso acima da rede
        roi_mask[min_y_roi:, :] = 255

        player_mask = self._create_player_mask(frame.shape, player_boxes)
        search_mask = self._create_search_mask(frame.shape, predicted_point)

        # 4. Combinação de Máscaras
        if self.kalman_initialized:
            yellow_or_motion = cv2.bitwise_or(yellow_mask, motion_mask)
            combined_mask = cv2.bitwise_and(yellow_or_motion, search_mask)
        else:
            # EXIGIR AMARELO E MOVIMENTO no início evita apanhar árvores/fundo solto
            combined_mask = cv2.bitwise_and(yellow_mask, motion_mask)

        combined_mask = cv2.bitwise_and(combined_mask, player_mask)
        combined_mask = cv2.bitwise_and(combined_mask, roi_mask)

        # 5. Filtrar Contornos
        contours, _ = cv2.findContours(
            combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1.0 or area > 120.0:  # Descarta manchas grandes do fundo
                continue

            (x, y), radius = cv2.minEnclosingCircle(contour)
            if radius < 0.5 or radius > 12.0:
                continue

            x_bound, y_bound, w_bound, h_bound = cv2.boundingRect(contour)
            aspect_ratio = float(w_bound) / float(h_bound)
            if aspect_ratio < 0.25 or aspect_ratio > 3.5:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue

            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.12:
                continue

            center = (int(x), int(y))

            # --- SEGUNDA BARREIRA CONTRA O JOGADOR ---
            is_player_noise = False
            if player_boxes:
                for box in player_boxes:
                    try:
                        bx1, by1, bx2, by2 = map(
                            int,
                            box[:4]
                            if isinstance(box, (list, tuple, np.ndarray))
                            else box.xyxy[0],
                        )
                        if (bx1 - 80) <= center[0] <= (bx2 + 30) and (
                            by1 - 30
                        ) <= center[1] <= (by2 + 50):
                            is_player_noise = True
                            break
                    except Exception:
                        pass
            if is_player_noise:
                continue
            # -----------------------------------------

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

        # 6. Escolher o melhor candidato
        ball_center = None
        world_pos = None

        if candidates:
            candidates.sort(key=lambda item: item["score"], reverse=True)

            for candidate in candidates:
                cand_center = candidate["center"]

                if (
                    self.last_confirmed_center is not None
                    and self.missed_frames < 3
                ):
                    dist_from_last = np.hypot(
                        cand_center[0] - self.last_confirmed_center[0],
                        cand_center[1] - self.last_confirmed_center[1],
                    )
                    if dist_from_last > 100.0:
                        continue

                if candidate["score"] >= 12.0:
                    ball_center = cand_center
                    world_pos = candidate["world_pos"]
                    break

        # 7. Atualização do Filtro de Kalman
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

            if self.static_frames_count >= 3:
                self.blocked_static_points.append(ball_center)
                self.reset_kalman()
                self.trajectory_pixels.clear()
                self.trajectory_world.clear()
                return None, None

            measurement = np.array(
                [[np.float32(ball_center[0])], [np.float32(ball_center[1])]],
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

            if court_model is not None:
                self.check_bounce(court_model)
        else:
            self.static_frames_count = 0
            self.missed_frames += 1
            self.trajectory_pixels.appendleft(None)
            self.trajectory_world.appendleft(None)

            if self.missed_frames > 15:
                self.reset_kalman()
                self.trajectory_pixels.clear()
                self.trajectory_world.clear()

        # 8. Janelas de Debug
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
        """Desenha a linha amarela de trajetória e o marcador da bola."""
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