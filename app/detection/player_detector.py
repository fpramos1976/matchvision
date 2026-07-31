import math
from ultralytics import YOLO


class PlayerDetector:

    def __init__(self, model_path="yolov8n.pt", confidence=0.5):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.person_class_id = 0

        # Histórico dos jogadores: {track_id: {"last_world_pos": (x, y), "total_distance": 0.0, "speed": 0.0}}
        self.players_stats = {}

    def detect_and_track(self, frame, homography, court_model, fps=30):
        """Detecta, rastreia e atualiza as estatísticas dos jogadores."""
        # O argumento persist=True habilita o ByteTRACK interno do Ultralytics
        results = self.model.track(
            frame,
            persist=True,
            verbose=False,
            conf=self.confidence,
        )[0]
        tracked_players = []

        if results.boxes is None or results.boxes.id is None:
            return tracked_players

        # Itera sobre as detecções rastreadas
        boxes = results.boxes.xyxy.cpu().numpy()
        track_ids = results.boxes.id.cpu().numpy().astype(int)
        classes = results.boxes.cls.cpu().numpy().astype(int)

        for box, track_id, cls in zip(boxes, track_ids, classes):
            if cls == self.person_class_id:
                x1, y1, x2, y2 = map(int, box)
                feet_x = int((x1 + x2) / 2)
                feet_y = y2

                # Converte para coordenadas reais em metros
                world_x, world_y = homography.transform_point_to_world(
                    (feet_x, feet_y)
                )

                # Filtra apenas quem está nos limites da quadra (+ margem)
                margin = 3.0
                if (
                    -margin <= world_x <= court_model.width + margin
                    and -margin <= world_y <= court_model.length + margin
                ):

                    # Inicializa estatísticas do jogador se for novo
                    if track_id not in self.players_stats:
                        self.players_stats[track_id] = {
                            "last_world_pos": (world_x, world_y),
                            "total_distance": 0.0,
                            "speed": 0.0,
                        }

                    # Atualiza distância e velocidade
                    prev_x, prev_y = self.players_stats[track_id][
                        "last_world_pos"
                    ]
                    dist_step = math.sqrt(
                        (world_x - prev_x) ** 2 + (world_y - prev_y) ** 2
                    )

                    # Filtra ruídos mínimos de vibração de câmera (< 5cm)
                    if dist_step > 0.05:
                        self.players_stats[track_id][
                            "total_distance"
                        ] += dist_step
                        # Velocidade em km/h = (metros/frame * fps) * 3.6
                        self.players_stats[track_id]["speed"] = (
                            dist_step * fps
                        ) * 3.6
                        self.players_stats[track_id]["last_world_pos"] = (
                            world_x,
                            world_y,
                        )
                    else:
                        # Se ficou parado, velocidade cai gradualmente
                        self.players_stats[track_id]["speed"] *= 0.8

                    tracked_players.append({
                        "id": track_id,
                        "bbox": (x1, y1, x2, y2),
                        "feet": (feet_x, feet_y),
                        "world_pos": (world_x, world_y),
                        "total_distance": self.players_stats[track_id][
                            "total_distance"
                        ],
                        "speed": self.players_stats[track_id]["speed"],
                    })

        return tracked_players