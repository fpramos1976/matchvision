import math
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

        # Histórico dos jogadores: {track_id: {"last_world_pos": (x, y), "total_distance": 0.0, "speed": 0.0}}
        self.players_stats = {}

    def _distance_outside_court(self, world_x, world_y, court_model):
        """Distância (m) do ponto até o retângulo da quadra; 0 se dentro."""
        dx = max(0.0, -world_x, world_x - court_model.width)
        dy = max(0.0, -world_y, world_y - court_model.length)
        return math.hypot(dx, dy)

    def detect_and_track(self, frame, homography, court_model, fps=30):
        """Detecta, rastreia e atualiza as estatísticas dos jogadores."""
        # O argumento persist=True habilita o ByteTRACK interno do Ultralytics
        results = self.model.track(
            frame,
            persist=True,
            verbose=False,
            conf=self.confidence,
            classes=[self.person_class_id],
            imgsz=self.image_size,
            tracker="bytetrack.yaml",
        )[0]
        tracked_players = []

        if results.boxes is None or results.boxes.id is None:
            return tracked_players

        # Itera sobre as detecções rastreadas
        boxes = results.boxes.xyxy.cpu().numpy()
        track_ids = results.boxes.id.cpu().numpy().astype(int)
        classes = results.boxes.cls.cpu().numpy().astype(int)
        confidences = results.boxes.conf.cpu().numpy()

        candidates = []
        for box, track_id, cls, conf in zip(boxes, track_ids, classes, confidences):
            if cls != self.person_class_id:
                continue

            x1, y1, x2, y2 = map(int, box)
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
            # jogadores já rastreados ganham bônus para não trocarem de
            # lugar com um gandula que passe perto.
            outside = self._distance_outside_court(world_x, world_y, court_model)
            priority = float(conf) - 0.1 * outside
            if track_id in self.players_stats:
                priority += 0.2

            candidates.append(
                (priority, track_id, (x1, y1, x2, y2), (feet_x, feet_y), (world_x, world_y))
            )

        candidates.sort(key=lambda item: item[0], reverse=True)
        if self.max_players:
            candidates = candidates[: self.max_players]

        for _, track_id, bbox, feet, (world_x, world_y) in candidates:
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
                # Salto impossível (troca de ID ou track reaparecendo em
                # outro lugar): reancora sem contabilizar distância.
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
                "bbox": bbox,
                "feet": feet,
                "world_pos": (smooth_x, smooth_y),
                "total_distance": stats["total_distance"],
                "speed": stats["speed"],
            })

        return tracked_players
