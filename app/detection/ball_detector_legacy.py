import collections
import cv2
import numpy as np


class BallDetector:

    def __init__(self, max_buffer=30):
        self.max_buffer = max_buffer

        self.trajectory_pixels = collections.deque(maxlen=max_buffer)
        self.trajectory_world = collections.deque(maxlen=max_buffer)

        self.previous_gray = None
        self.prev_gray_2 = None

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
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 12, 0], [0, 0, 0, 12]],
                dtype=np.float32,
            )
            * 0.05
        )
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 2.0
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32) * 10.0

    def reset_kalman(self):
        """Reinicializa o Kalman quando a bola fica perdida ou estática."""
        self.kalman_initialized = False
        self.missed_frames = 0
        self.static_frames_count = 0
        self.last_confirmed_center = None

        self.kalman.statePre = np.zeros((4, 1), dtype=np.float32)
        self.kalman.statePost = np.zeros((4, 1), dtype=np.float32)

    def check_bounce(self, court_model):
        """Analisa a variação de vetor para detetar ressaltos reais."""
        valid_points = [p for p in self.trajectory_world if p is not None]
        if len(valid_points) < 4:
            return None

        p1, p2, p3, p4 = (
            valid_points[0],
            valid_points[1],
            valid_points[2],
            valid_points[3],
        )

        dy1 = p2[1] - p1[1]
        dy2 = p3[1] - p2[1]
        dy3 = p4[1] - p3[1]

        if (dy1 * dy2 < 0) or (dy2 * dy3 < 0):
            bounce_point = p2

            if self.bounces:
                last_b_pos, _ = self.bounces[-1]
                dist = np.hypot(
                    bounce_point[0] - last_b_pos[0],
                    bounce_point[1] - last_b_pos[1],
                )
                if dist < 1.0:
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
        """Oculta apenas o corpo estrito do jogador, permitindo apanhar bolas muito próximas."""
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
                pad = 5
                px1 = max(0, x1 - pad)
                py1 = max(0, y1 - pad)
                px2 = min(width, x2 + pad)
                py2 = min(height, y2 + pad)

                player_mask[py1:py2, px1:px2] = 0
            except Exception:
                continue

        return player_mask

    def _create_yellow_mask(self, frame):
        """Mascara HSV com tolerância alargada para capturar amarelo em movimento."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array([12, 20, 30], dtype=np.uint8)
        upper_yellow = np.array([75, 255, 255], dtype=np.uint8)

        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        yellow_mask = cv2.dilate(yellow_mask, kernel, iterations=1)

        return yellow_mask

    def _create_motion_mask(self, gray):
        """Acumulador de movimento em 3 frames consecutivos para revelar pontos minúsculos."""
        if self.previous_gray is None:
            self.previous_gray = gray.copy()
            self.prev_gray_2 = gray.copy()
            return np.zeros_like(gray, dtype=np.uint8)

        diff1 = cv2.absdiff(gray, self.previous_gray)
        diff2 = cv2.absdiff(self.previous_gray, self.prev_gray_2)

        # Combina o movimento entre t, t-1 e t-2
        motion_combined = cv2.bitwise_or(diff1, diff2)
        _, motion_mask = cv2.threshold(motion_combined, 8, 255, cv2.THRESH_BINARY)

        # Filtro de sombras escuras do piso
        _, bright_mask = cv2.threshold(gray, 65, 255, cv2.THRESH_BINARY)
        motion_mask = cv2.bitwise_and(motion_mask, bright_mask)

        self.prev_gray_2 = self.previous_gray.copy()
        self.previous_gray = gray.copy()
        return motion_mask

    def _create_search_mask(self, image_shape, predicted_point):
        """Cria uma região circular de pesquisa em redor da previsão do Kalman."""
        height, width = image_shape[:2]
        search_mask = np.full((height, width), 255, dtype=np.uint8)

        if predicted_point is None:
            return search_mask

        predicted_x, predicted_y = predicted_point
        if not (0 <= predicted_x < width and 0 <= predicted_y < height):
            return search_mask

        radius = min(150, 70 + (self.missed_frames * 12))
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

            # Expande as margens longitudinais (Y) para não rejeitar a bola quando voa alta
            margin_x = 2.5
            margin_y = 5.0  # Alargado para absorver o erro de paralaxe da altura da bola

            inside = (
                -margin_x <= world_x <= (court_model.width + margin_x)
                and -margin_y <= world_y <= (court_model.length + margin_y)
            )

            return inside, (world_x, world_y)

    def _is_point_blacklisted(self, center):
        for blocked_pt in self.blocked_static_points:
            dist = np.hypot(center[0] - blocked_pt[0], center[1] - blocked_pt[1])
            if dist < 20.0:
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
        is_far_field,
    ):
        score = 0.0

        # Aceita áreas e raios substancialmente mais pequenos
        ideal_radius = 2.5 if is_far_field else 4.0
        radius_diff = abs(radius - ideal_radius)
        score += max(0.0, 25.0 - (radius_diff * 5.0))

        score += min(15.0, circularity * 15.0)

        if motion_value > 0:
            score += 30.0

        if yellow_value > 0:
            score += 20.0
        elif is_far_field:
            # Recompensa candidatos distantes mesmo sem sinal de cor amarela
            score += 15.0

        if predicted_point is not None:
            distance = float(
                np.hypot(
                    center[0] - predicted_point[0],
                    center[1] - predicted_point[1],
                )
            )
            score += max(0.0, 40.0 - (distance * 0.25))

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

        # ROI de contexto
        roi_mask = np.zeros((height, width), dtype=np.uint8)
        min_y_roi = int(height * 0.25)
        roi_mask[min_y_roi:, :] = 255

        player_mask = self._create_player_mask(frame.shape, player_boxes)
        search_mask = self._create_search_mask(frame.shape, predicted_point)

        # 4. Combinação de Máscaras (Privilegia o movimento)
        combined_mask = cv2.bitwise_and(motion_mask, player_mask)
        combined_mask = cv2.bitwise_and(combined_mask, roi_mask)

        if self.kalman_initialized:
            combined_mask = cv2.bitwise_and(combined_mask, search_mask)

        # 5. Filtrar Contornos de Múltiplos Tamanhos
        contours, _ = cv2.findContours(
            combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            # Permite objetos de 0.2px até 150px
            if area < 0.2 or area > 150.0:
                continue

            (x, y), radius = cv2.minEnclosingCircle(contour)
            if radius < 0.2 or radius > 15.0:
                continue

            center = (int(x), int(y))
            is_far_field = center[1] < int(height * 0.55)

            # Contorno de linhas horizontais extensas (ex: rede)
            x_bound, y_bound, w_bound, h_bound = cv2.boundingRect(contour)
            aspect_ratio = float(w_bound) / float(max(1, h_bound))
            if aspect_ratio > 3.5:
                continue

            perimeter = cv2.arcLength(contour, True)
            circularity = (
                (4.0 * np.pi * area / (perimeter * perimeter))
                if perimeter > 0
                else 0.5
            )

            # Permite contornos mais desfocados/esticados
            if circularity < 0.05:
                continue

            # Elimina contornos dentro da caixa do jogador
            is_player_body = False
            if player_boxes:
                for box in player_boxes:
                    try:
                        bx1, by1, bx2, by2 = map(
                            int,
                            box[:4]
                            if isinstance(box, (list, tuple, np.ndarray))
                            else box.xyxy[0],
                        )
                        if bx1 <= center[0] <= bx2 and by1 <= center[1] <= by2:
                            is_player_body = True
                            break
                    except Exception:
                        pass
            if is_player_body:
                continue

            if self._is_point_blacklisted(center):
                continue

            inside_court, candidate_world = self._is_inside_court(
                center, homography, court_model
            )
            if not inside_court:
                continue

            center_x = min(width - 1, max(0, center[0]))
            center_y = min(height - 1, max(0, center[1]))

            yellow_val = int(yellow_mask[center_y, center_x])
            motion_val = int(motion_mask[center_y, center_x])

            score = self._score_candidate(
                center,
                radius,
                area,
                circularity,
                predicted_point,
                yellow_val,
                motion_val,
                is_far_field,
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

        # 6. Escolher o melhor candidato
        ball_center = None
        world_pos = None

        if candidates:
            candidates.sort(key=lambda item: item["score"], reverse=True)

            for candidate in candidates:
                cand_center = candidate["center"]

                if (
                    self.last_confirmed_center is not None
                    and self.missed_frames < 4
                ):
                    dist_from_last = np.hypot(
                        cand_center[0] - self.last_confirmed_center[0],
                        cand_center[1] - self.last_confirmed_center[1],
                    )
                    if dist_from_last > 140.0:
                        continue

                # Aceita candidatos com pontuação mínima a partir de 3.0
                if candidate["score"] >= 3.0:
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

            if self.static_frames_count >= 4:
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

            if self.missed_frames > 20:
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
                    radius + 2,
                    (0, 255, 255),
                    1,
                )

            for pt in self.blocked_static_points:
                cv2.circle(debug_candidates, pt, 10, (0, 0, 0), 2)

            if predicted_point is not None:
                cv2.circle(
                    debug_candidates, predicted_point, 8, (255, 0, 255), 2
                )

            if ball_center is not None:
                cv2.circle(debug_candidates, ball_center, 7, (0, 0, 255), 2)

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