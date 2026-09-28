import math

import numpy as np
from ultralytics import YOLO


class PlayerDetector:

    def __init__(
        self,
        model_path="yolov8n.pt",
        confidence=0.25,
        max_players=2,
        image_size=1280,
        smoothing=0.4,
        max_speed_kmh=40.0,
        far_confidence=0.15,
        far_image_size=640,
    ):
        self.model = YOLO(model_path)
        # Confiança mais baixa + imagem maior: o jogador do fundo da quadra
        # ocupa poucos pixels e, com conf=0.5 em 640px, o yolov8n o perdia
        # com frequência. Os falsos positivos extras (público, gandulas,
        # juiz) são descartados pelo filtro de quadra + max_players.
        self.confidence = confidence
        self.image_size = image_size
        self.max_players = max_players
        self.person_class_id = 0

        # Segunda passada no recorte da metade distante da quadra. Mesmo
        # em 1280px o jogador do fundo tem ~20-30px de altura e o YOLO
        # não o encontra; recortando só aquela região e ampliando-a para
        # far_image_size, ele passa a ocupar ~5x mais pixels na rede.
        self.far_confidence = far_confidence
        self.far_image_size = far_image_size

        # Suavização exponencial da posição no mundo (0 = sem suavizar,
        # 1 = congela). Reduz a tremulação da caixa, que inflava a
        # distância percorrida e a velocidade.
        self.smoothing = smoothing
        # Saltos acima desta velocidade são troca de ID / erro de caixa,
        # não movimento real de um tenista.
        self.max_speed_kmh = max_speed_kmh

        # Margens em metros ao redor da quadra. No eixo longitudinal a
        # margem é maior porque os jogadores ficam vários metros atrás da
        # linha de base (a área de fundo tem ~6.4 m em quadras oficiais).
        self.margin_x = 3.0
        self.margin_y = 6.0

        # Rastreamento próprio por posição na quadra (em metros). Com 2-4
        # jogadores que raramente se cruzam, casar pela posição no mundo é
        # mais estável que o ByteTrack em pixels e funciona também para as
        # detecções vindas do recorte do fundo.
        self.frame_number = 0
        self.next_track_id = 1
        self.max_track_age = 60  # frames sem detecção antes de descartar
        self.tracks = {}  # {track_id: {"world": (x, y), "last_seen": frame}}

        # Histórico dos jogadores: {track_id: {"last_world_pos": (x, y), "total_distance": 0.0, "speed": 0.0}}
        self.players_stats = {}

    def _distance_outside_court(self, world_x, world_y, court_model):
        """Distância (m) do ponto até o retângulo da quadra; 0 se dentro."""
        dx = max(0.0, -world_x, world_x - court_model.width)
        dy = max(0.0, -world_y, world_y - court_model.length)
        return math.hypot(dx, dy)

    def _run_model(self, image, confidence, image_size, offset=(0, 0)):
        """Roda o YOLO e devolve [(x1, y1, x2, y2, conf)] em coordenadas do frame."""
        results = self.model.predict(
            image,
            verbose=False,
            conf=confidence,
            classes=[self.person_class_id],
            imgsz=image_size,
        )[0]
        if results.boxes is None or len(results.boxes) == 0:
            return []

        ox, oy = offset
        boxes = results.boxes.xyxy.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        return [
            (int(b[0]) + ox, int(b[1]) + oy, int(b[2]) + ox, int(b[3]) + oy, float(c))
            for b, c in zip(boxes, confidences)
        ]

    def _far_court_region(self, frame_shape, homography, court_model):
        """Retângulo (x1, y1, x2, y2) em pixels cobrindo a metade da quadra
        mais distante da câmera, incluindo a área atrás da linha de base e
        espaço acima dela para o corpo do jogador."""
        height, width = frame_shape[:2]
        try:
            # Descobre qual linha de base (y=0 ou y=length) está mais longe
            # da câmera: é a que aparece mais acima na imagem.
            y0 = homography.transform_point((court_model.width / 2, 0.0))[1]
            y_len = homography.transform_point(
                (court_model.width / 2, court_model.length)
            )[1]
            if y0 < y_len:
                far_y, behind_y = 0.0, -self.margin_y
            else:
                far_y, behind_y = court_model.length, court_model.length + self.margin_y
            # Da área atrás da linha de base até a linha de saque: perto da
            # rede o jogador já aparece grande o bastante no frame inteiro,
            # e um recorte menor é ampliado mais pelo YOLO.
            service_y = far_y + (6.40 if far_y == 0.0 else -6.40)

            world_corners = [
                (-self.margin_x, behind_y),
                (court_model.width + self.margin_x, behind_y),
                (court_model.width + 1.0, service_y),
                (-1.0, service_y),
            ]
            pts = np.array(
                [homography.transform_point(p) for p in world_corners], dtype=np.float32
            )

            left = homography.transform_point((0.0, far_y))
            right = homography.transform_point((court_model.width, far_y))
        except Exception:
            return None

        # Altura de uma pessoa (~2.2 m) na escala da linha de base distante
        px_per_meter = math.hypot(right[0] - left[0], right[1] - left[1]) / court_model.width
        body_height = 2.5 * px_per_meter

        x1 = int(max(0, pts[:, 0].min()))
        x2 = int(min(width, pts[:, 0].max()))
        y1 = int(max(0, pts[:, 1].min() - body_height))
        # Folga embaixo: com a câmera baixa, erros pequenos de calibração
        # deslocam bastante a linha distante na imagem.
        y2 = int(min(height, pts[:, 1].max() + 0.5 * body_height))

        if x2 - x1 < 32 or y2 - y1 < 32:
            return None
        # Se o recorte já é quase o frame todo, ampliar não ajuda
        if (x2 - x1) * (y2 - y1) > 0.6 * width * height:
            return None
        return x1, y1, x2, y2

    @staticmethod
    def _iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / float(area_a + area_b - inter)

    @staticmethod
    def _containment(inner, outer):
        """Fração da área de `inner` que está dentro de `outer`."""
        ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
        ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area = max(1, (inner[2] - inner[0]) * (inner[3] - inner[1]))
        return inter / float(area)

    def _merge_detections(self, detections):
        """NMS simples entre as detecções do frame inteiro e do recorte."""
        detections = sorted(detections, key=lambda d: d[4], reverse=True)
        kept = []
        for det in detections:
            if all(self._iou(det, k) < 0.4 for k in kept):
                kept.append(det)
        return kept

    def _assign_track_ids(self, candidates, fps):
        """Casa cada candidato com o track mais próximo (em metros)."""
        pairs = []
        for ci, cand in enumerate(candidates):
            wx, wy = cand["world"]
            for tid, track in self.tracks.items():
                frames_gone = self.frame_number - track["last_seen"]
                max_jump = 1.5 + (self.max_speed_kmh / 3.6) * frames_gone / fps
                dist = math.hypot(wx - track["world"][0], wy - track["world"][1])
                if dist <= max_jump:
                    pairs.append((dist, ci, tid))

        pairs.sort()
        assigned, used_tracks = {}, set()
        for _, ci, tid in pairs:
            if ci in assigned or tid in used_tracks:
                continue
            assigned[ci] = tid
            used_tracks.add(tid)

        for ci, cand in enumerate(candidates):
            tid = assigned.get(ci)
            if tid is None:
                tid = self.next_track_id
                self.next_track_id += 1
            cand["id"] = tid
            self.tracks[tid] = {"world": cand["world"], "last_seen": self.frame_number}

        for tid in [
            t
            for t, tr in self.tracks.items()
            if self.frame_number - tr["last_seen"] > self.max_track_age
        ]:
            del self.tracks[tid]

    def detect_and_track(self, frame, homography, court_model, fps=30):
        """Detecta, rastreia e atualiza as estatísticas dos jogadores."""
        self.frame_number += 1

        detections = self._run_model(frame, self.confidence, self.image_size)

        region = self._far_court_region(frame.shape, homography, court_model)
        if region is not None:
            x1, y1, x2, y2 = region
            for det in self._run_model(
                frame[y1:y2, x1:x2],
                self.far_confidence,
                self.far_image_size,
                offset=(x1, y1),
            ):
                # Pessoa cortada pela borda inferior do recorte é o tronco
                # de alguém mais perto da câmera: os "pés" seriam falsos.
                if det[3] >= y2 - 2 and y2 < frame.shape[0]:
                    continue
                # Já coberta por uma detecção do frame inteiro
                if any(self._containment(det, full) > 0.6 for full in detections):
                    continue
                # A rede costuma esconder as pernas do jogador do fundo e a
                # caixa termina na faixa da rede, jogando os "pés" metros à
                # frente. Uma pessoa em pé tem altura ~2.4x a largura: se a
                # caixa for mais baixa que isso, estende até onde os pés
                # deveriam estar.
                bx1, by1, bx2, by2, bconf = det
                expected_h = 2.4 * (bx2 - bx1)
                if (by2 - by1) < 0.8 * expected_h:
                    by2 = int(min(frame.shape[0] - 1, by1 + expected_h))
                    det = (bx1, by1, bx2, by2, bconf)
                detections.append(det)

        detections = self._merge_detections(detections)

        candidates = []
        for x1, y1, x2, y2, conf in detections:
            feet_x = int((x1 + x2) / 2)
            feet_y = y2

            # Converte para coordenadas reais em metros
            world_x, world_y = homography.transform_point_to_world(
                (feet_x, feet_y)
            )

            # Filtra apenas quem está nos limites da quadra (+ margem)
            if not (
                -self.margin_x <= world_x <= court_model.width + self.margin_x
                and -self.margin_y <= world_y <= court_model.length + self.margin_y
            ):
                continue

            # Prioriza quem está dentro/perto da quadra e com alta confiança;
            # quem está perto de um jogador já rastreado ganha bônus para
            # não trocar de lugar com um gandula que passe perto.
            outside = self._distance_outside_court(world_x, world_y, court_model)
            priority = conf - 0.1 * outside
            if any(
                math.hypot(world_x - t["world"][0], world_y - t["world"][1]) < 2.0
                for t in self.tracks.values()
            ):
                priority += 0.2

            candidates.append({
                "priority": priority,
                "bbox": (x1, y1, x2, y2),
                "feet": (feet_x, feet_y),
                "world": (world_x, world_y),
            })

        candidates.sort(key=lambda c: c["priority"], reverse=True)
        if self.max_players:
            candidates = candidates[: self.max_players]

        self._assign_track_ids(candidates, fps)

        tracked_players = []
        for cand in candidates:
            track_id = cand["id"]
            world_x, world_y = cand["world"]

            # Inicializa estatísticas do jogador se for novo
            if track_id not in self.players_stats:
                self.players_stats[track_id] = {
                    "last_world_pos": (world_x, world_y),
                    "smoothed_pos": (world_x, world_y),
                    "total_distance": 0.0,
                    "speed": 0.0,
                }

            stats = self.players_stats[track_id]

            # Suaviza a posição antes de medir deslocamento
            prev_sx, prev_sy = stats["smoothed_pos"]
            alpha = self.smoothing
            smooth_x = alpha * prev_sx + (1.0 - alpha) * world_x
            smooth_y = alpha * prev_sy + (1.0 - alpha) * world_y

            # Velocidade instantânea em km/h = (metros/frame * fps) * 3.6
            step = math.hypot(smooth_x - prev_sx, smooth_y - prev_sy)
            instant_speed = step * fps * 3.6

            if instant_speed > self.max_speed_kmh:
                # Salto impossível (track reaparecendo em outro lugar):
                # reancora sem contabilizar distância.
                smooth_x, smooth_y = world_x, world_y
                stats["last_world_pos"] = (smooth_x, smooth_y)
            else:
                stats["speed"] = 0.7 * stats["speed"] + 0.3 * instant_speed

                # Filtra ruídos mínimos de vibração de câmera (< 5cm)
                prev_x, prev_y = stats["last_world_pos"]
                dist_step = math.hypot(smooth_x - prev_x, smooth_y - prev_y)
                if dist_step > 0.05:
                    stats["total_distance"] += dist_step
                    stats["last_world_pos"] = (smooth_x, smooth_y)

            stats["smoothed_pos"] = (smooth_x, smooth_y)

            tracked_players.append({
                "id": track_id,
                "bbox": cand["bbox"],
                "feet": cand["feet"],
                "world_pos": (smooth_x, smooth_y),
                "total_distance": stats["total_distance"],
                "speed": stats["speed"],
            })

        return tracked_players
