import collections
import cv2
import numpy as np


class BallDetector:

    def __init__(self, max_buffer=30):
        self.max_buffer = max_buffer

        self.trajectory_pixels = collections.deque(maxlen=max_buffer)
        self.trajectory_world = collections.deque(maxlen=max_buffer)
        self.confirmed_positions_history = collections.deque(maxlen=6)

        self.previous_gray = None
        self.prev_gray_2 = None

        # --- ESTADOS E CONTADORES ---
        self.kalman_initialized = False
        self.missed_frames = 0
        self.static_frames_count = 0
        self.last_confirmed_center = None
        self.frame_number = 0
        self.show_debug = True
        self.verbose_logging = True

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
        self.confirmed_positions_history.clear()

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
        """Mascara HSV calibrada com pixels reais da bola do vídeo
        tennis_2.mp4 (medidos com tests/inspect_ball_color.py):
        H entre 15-22, S entre 106-138, V entre 140-188 sob luz solar
        direta.

        O brilho (V) é o canal mais discriminante: ao medir pontos do
        fundo (prédio, ramos secos, terra) que caem na mesma faixa de
        matiz/saturação da bola, o maior valor de V encontrado entre
        eles foi 72 - bem abaixo do menor V medido na bola (140). Por
        isso o limite inferior de V foi definido em 130: mantém uma
        margem de segurança confortável dos dois lados, sem exigir
        exatamente o valor mínimo medido na bola.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array([10, 80, 130], dtype=np.uint8)
        upper_yellow = np.array([30, 255, 255], dtype=np.uint8)

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

        # Pontos cujo pixel está ACIMA da linha mais alta calibrada (ou seja,
        # acima de todos os 4 cantos usados na calibração) representam,
        # necessariamente, algo no ar acima da quadra - não existe chão ali
        # para a homografia interpretar. Nessa faixa, a homografia plana
        # extrapola e pode devolver coordenadas de mundo muito distantes
        # mesmo que a bola esteja, na realidade, sobre a quadra.
        # Por isso não aplicamos o filtro geométrico aqui: confiamos nos
        # outros filtros (cor, movimento, continuidade do Kalman) para
        # decidir se o candidato é válido.
        calibrated_min_y = min(
            homography.image_points.top_left[1],
            homography.image_points.top_right[1],
            homography.image_points.bottom_right[1],
            homography.image_points.bottom_left[1],
        )

        if center[1] < calibrated_min_y:
            return True, (world_x, world_y)

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
            # O bônus de campo distante só se aplica quando também há
            # sinal de cor - sem isso, ruído de fundo (folhas, sombras)
            # em campo distante ganhava pontos apenas por estar longe,
            # o que causava falsos positivos sem nenhuma relação com a
            # cor real da bola.
            if is_far_field:
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

    def _is_trajectory_erratic(self, candidate_center):
        """Verifica se aceitar candidate_center resultaria numa trajetória
        errática (ziguezague), típica de ruído de vegetação balançando,
        em vez de um voo real de bola de tênis.

        Compara o deslocamento líquido (ponto inicial -> ponto final da
        janela recente) com a distância total percorrida passo a passo.
        Uma bola em voo real tem eficiência alta (~0.6-1.0): ela vai
        consistentemente para algum lugar. Ruído oscilante (folhas ao
        vento) tem eficiência baixa: percorre bastante distância total
        sem se afastar muito do ponto de partida.
        """
        if len(self.confirmed_positions_history) < 4:
            return False

        points = list(self.confirmed_positions_history) + [candidate_center]

        total_path_length = sum(
            np.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1])
            for i in range(1, len(points))
        )

        if total_path_length < 1e-6:
            return False

        net_displacement = np.hypot(
            points[-1][0] - points[0][0], points[-1][1] - points[0][1]
        )

        efficiency = net_displacement / total_path_length
        return efficiency < 0.3

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

        # ROI de contexto: quando temos a homografia, calculamos o corte a
        # partir da linha mais alta calibrada, com folga extra para cima
        # (permite bolas voando acima do ponto mais distante calibrado,
        # normal em saques). Sem homografia, usamos o valor fixo antigo
        # como fallback.
        if homography is not None:
            calibrated_min_y = min(
                homography.image_points.top_left[1],
                homography.image_points.top_right[1],
                homography.image_points.bottom_right[1],
                homography.image_points.bottom_left[1],
            )
            roi_top = max(0, int(calibrated_min_y - height * 0.6))
        else:
            roi_top = int(height * 0.25)

        roi_mask = np.zeros((height, width), dtype=np.uint8)
        roi_mask[roi_top:, :] = 255

        player_mask = self._create_player_mask(frame.shape, player_boxes)
        search_mask = self._create_search_mask(frame.shape, predicted_point)

        # 4. Combinação de Máscaras
        # A bola pode estar momentaneamente quase parada (ápice do toss
        # no saque), quando o motion_mask não gera sinal suficiente.
        # Por isso um candidato pode entrar por movimento OU por cor
        # amarela forte - a cor sozinha também qualifica um contorno
        # para ser avaliado, mas o score final (_score_candidate)
        # ainda decide se ele é aceito.
        motion_or_color_mask = cv2.bitwise_or(motion_mask, yellow_mask)
        combined_mask = cv2.bitwise_and(motion_or_color_mask, player_mask)
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

            if motion_val == 0 and yellow_val > 0:
                # is_far_field (y < 55% da altura) é muito permissivo para
                # este filtro - ele inclui toda a altura da rede, que é
                # exatamente onde vimos elementos fixos (ilhós, parafusos)
                # sendo confundidos com a bola. Um toss de saque parado só
                # é plausível bem acima da linha mais alta calibrada, não
                # em qualquer ponto da metade superior do frame.
                if homography is not None:
                    calibrated_min_y = min(
                        homography.image_points.top_left[1],
                        homography.image_points.top_right[1],
                        homography.image_points.bottom_right[1],
                        homography.image_points.bottom_left[1],
                    )
                    high_altitude_threshold = calibrated_min_y - height * 0.25
                else:
                    high_altitude_threshold = height * 0.15

                if center[1] > high_altitude_threshold:
                    continue
                if circularity < 0.55 or radius > 6.0:
                    continue

            # Um candidato que só tem sinal de cor, sem nenhum movimento,
            # é o caso mais explorado por elementos estáticos e amarelados
            # do fundo (faixas de propaganda, fita da rede sob luz forte).
            # Para esse caso específico, exigimos geometria muito mais
            # estrita - quase um círculo perfeito e raio pequeno - já que
            # a bola raramente fica perfeitamente parada por vários frames.
            if motion_val == 0 and yellow_val > 0:
                if circularity < 0.55 or radius > 6.0:
                    continue


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

        # 6. Escolher o melhor candidato (agora também exige trajetória plausível)
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

                # Aceita candidatos com pontuação mínima a partir de 3.0,
                # desde que a trajetória resultante não seja errática.
                if candidate["score"] >= 3.0:
                    if self._is_trajectory_erratic(cand_center):
                        continue
                    ball_center = cand_center
                    world_pos = candidate["world_pos"]
                    break

        # --- LOG DE DIAGNÓSTICO (temporário) ---
        if self.verbose_logging:
            top_scores = sorted(
                (round(c["score"], 1) for c in candidates), reverse=True
            )[:3]
            best_candidate_info = "n/a"
            if candidates:
                best = max(candidates, key=lambda c: c["score"])
                bx, by = best["center"]
                by_clamped = min(height - 1, max(0, by))
                bx_clamped = min(width - 1, max(0, bx))
                best_yellow = int(yellow_mask[by_clamped, bx_clamped])
                best_motion = int(motion_mask[by_clamped, bx_clamped])
                best_candidate_info = (
                    f"pos=({bx},{by}) yellow={best_yellow} motion={best_motion}"
                )
            print(
                f"[frame {self.frame_number}] "
                f"contornos_brutos={len(contours)} "
                f"candidatos_validos={len(candidates)} "
                f"top_scores={top_scores} "
                f"ball_center={ball_center} "
                f"melhor_candidato=({best_candidate_info}) "
                f"kalman_ativo={self.kalman_initialized} "
                f"missed_frames={self.missed_frames} "
                f"blacklist={len(self.blocked_static_points)}"
            )

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

            if self.static_frames_count >= 2:
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
            self.confirmed_positions_history.append(ball_center)

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