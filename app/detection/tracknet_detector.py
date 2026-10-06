"""Detector de bola baseado no TrackNet (rede neural treinada para bola de
tênis pequena e borrada).

A rede recebe 3 frames consecutivos (atual, anterior e o de antes) em
640x360 e devolve, para cada pixel, a probabilidade de ser a bola. Os pesos
em models/tracknet.pt são os publicados em
https://github.com/yastrebksv/TrackNet (treinados em transmissões de TV
1280x720 a 30 fps); a arquitetura abaixo reproduz a do artigo com os
mesmos nomes de camadas para o state_dict carregar sem conversão.
"""

import collections

import cv2
import numpy as np
import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.block(x)


class _TrackNet(nn.Module):
    """Encoder-decoder estilo VGG: 9 canais de entrada (3 frames BGR) e
    256 canais de saída (classes de intensidade do heatmap por pixel)."""

    def __init__(self, out_channels=256):
        super().__init__()
        c = _ConvBlock
        self.conv1, self.conv2 = c(9, 64), c(64, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3, self.conv4 = c(64, 128), c(128, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5, self.conv6, self.conv7 = c(128, 256), c(256, 256), c(256, 256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv8, self.conv9, self.conv10 = c(256, 512), c(512, 512), c(512, 512)
        self.ups1 = nn.Upsample(scale_factor=2)
        self.conv11, self.conv12, self.conv13 = c(512, 256), c(256, 256), c(256, 256)
        self.ups2 = nn.Upsample(scale_factor=2)
        self.conv14, self.conv15 = c(256, 128), c(128, 128)
        self.ups3 = nn.Upsample(scale_factor=2)
        self.conv16, self.conv17 = c(128, 64), c(64, 64)
        self.conv18 = c(64, out_channels)

    def forward(self, x):
        x = self.pool1(self.conv2(self.conv1(x)))
        x = self.pool2(self.conv4(self.conv3(x)))
        x = self.pool3(self.conv7(self.conv6(self.conv5(x))))
        x = self.ups1(self.conv10(self.conv9(self.conv8(x))))
        x = self.ups2(self.conv13(self.conv12(self.conv11(x))))
        x = self.ups3(self.conv15(self.conv14(x)))
        return self.conv18(self.conv17(self.conv16(x)))


class TrackNetBallDetector:
    """Mesma interface usada pelo VideoProcessor: detect() devolve
    (centro_em_pixels, posição_no_mundo) e draw_ball_trail() desenha o
    rastro."""

    INPUT_WIDTH = 640
    INPUT_HEIGHT = 360
    RESCUE_MIN_SATURATION = 100
    # Saturação mediana mínima da mancha (a bola minúscula oscila entre ~110 e ~200)
    RESCUE_MIN_BLOB_SATURATION = 110
    # Margem (px em vídeo de 1920 de largura) em volta da caixa do jogador
    PLAYER_ZONE_PAD = 15
    # Diferença média de brilho entre quadros consecutivos sobre a mancha:
    # bola em voo ~25-40; capim/folhagem parados ~0-2
    RESCUE_MIN_MOTION = 8

    def __init__(
        self,
        model_path="models/tracknet.pt",
        fps=30,
        max_buffer=30,
        heatmap_threshold=127,
        max_jump=100,
        device=None,
        color_rescue=True,
    ):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"  # GPU dos Macs com Apple Silicon
            else:
                device = "cpu"
        self.device = device
        self.model = _TrackNet()
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

        # Três frames consecutivos (t, t-1, t-2), como no treino. Testamos
        # espaçar os frames em vídeos de 60 fps (t, t-2, t-4): no
        # trecho_hd.mp4 isso gerou 3 detecções erradas contra 0 com frames
        # vizinhos, para o mesmo número de acertos.
        self.frame_step = 1
        self.frames = collections.deque(maxlen=2 * self.frame_step + 1)

        self.heatmap_threshold = heatmap_threshold
        # Saltos maiores que isto entre detecções seguidas são tratados
        # como falso positivo. O valor é em pixels de um vídeo 1280 px de
        # largura e é escalado para a resolução real.
        self.max_jump = max_jump

        # Buracos curtos na trajetória (a rede perde a bola colada ao
        # jogador) são preenchidos por interpolação quando a bola
        # reaparece perto. ~1/6 s: 10 frames a 60 fps, 4 a 24 fps. No
        # trecho_hd.mp4 isso levou de 27 para 38 acertos em 46 quadros
        # conferidos, sem nenhuma posição errada.
        self.max_gap = max(1, int(round(fps / 6.0)))

        self.trajectory_pixels = collections.deque(maxlen=max_buffer)
        self.trajectory_world = collections.deque(maxlen=max_buffer)
        self.last_center = None
        self.missed_frames = 0

        # Resgate por cor: quando a rede perde a bola (fundo escuro, bola
        # grande e colada ao jogador, fora do domínio de TV em que foi
        # treinada), procura uma mancha amarela compacta só em volta da
        # posição prevista pelo movimento. A janela e a sequência máxima de
        # resgates impedem que o rastro "grude" numa raquete amarela.
        self.color_rescue = color_rescue
        self.rescue_window = max(2, int(round(fps / 3.0)))
        self.max_rescue_streak = max(2, int(round(fps / 4.0)))
        self.rescue_streak = 0
        self.velocity = (0.0, 0.0)

    def _heatmap(self):
        step = self.frame_step
        current, previous, before = self.frames[-1], self.frames[-1 - step], self.frames[0]
        resized = [
            cv2.resize(f, (self.INPUT_WIDTH, self.INPUT_HEIGHT)) for f in (current, previous, before)
        ]
        stacked = np.concatenate(resized, axis=2).astype(np.float32) / 255.0
        tensor = torch.from_numpy(np.ascontiguousarray(stacked.transpose(2, 0, 1)))[None]
        with torch.no_grad():
            out = self.model(tensor.to(self.device))
        # Classe de intensidade (0-255) mais provável em cada pixel
        return out.argmax(dim=1)[0].byte().cpu().numpy()

    def _ball_from_heatmap(self, heatmap):
        _, binary = cv2.threshold(heatmap, self.heatmap_threshold, 255, cv2.THRESH_BINARY)
        count, _, stats, centroids = cv2.connectedComponentsWithStats(binary)
        if count <= 1:
            return None
        # Maior mancha do heatmap = posição mais provável da bola
        best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return centroids[best]

    def detect(self, frame, homography=None, court_model=None, player_boxes=None):
        self.frames.append(frame)
        center = None

        if len(self.frames) == self.frames.maxlen:
            spot = self._ball_from_heatmap(self._heatmap())
            if spot is not None:
                height, width = frame.shape[:2]
                center = (
                    int(round(spot[0] * width / self.INPUT_WIDTH)),
                    int(round(spot[1] * height / self.INPUT_HEIGHT)),
                )

        max_jump = self.max_jump * frame.shape[1] / 1280.0
        if center is not None and self.last_center is not None and self.missed_frames < 4:
            jump = np.hypot(center[0] - self.last_center[0], center[1] - self.last_center[1])
            if jump > max_jump * (1 + self.missed_frames):
                center = None

        rescued = False
        if center is None and self.color_rescue:
            center = self._rescue_by_color(frame, player_boxes)
            rescued = center is not None

        world_pos = None
        if center is not None:
            world_pos = self._to_world(center, homography)
            self._fill_gap(center, homography, max_jump)
            self._update_velocity(center)
            self.last_center = center
            self.missed_frames = 0
            self.rescue_streak = self.rescue_streak + 1 if rescued else 0
        else:
            self.missed_frames += 1

        self.trajectory_pixels.appendleft(center)
        self.trajectory_world.appendleft(world_pos)
        return center, world_pos

    def _in_player_zone(self, point, player_boxes, scale):
        """True se o ponto está dentro da caixa de um jogador (com uma
        pequena margem). Uma raquete amarela tem cor parecida com a da
        bola e fica colada ao corpo; a bola colada ao jogador continua
        sendo coberta pela própria rede."""
        pad = self.PLAYER_ZONE_PAD * scale
        for box in player_boxes or []:
            try:
                if hasattr(box, "xyxy"):
                    box = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = (float(v) for v in box[:4])
            except (AttributeError, TypeError, ValueError, IndexError):
                continue
            if x1 - pad <= point[0] <= x2 + pad and y1 - pad <= point[1] <= y2 + pad:
                return True
        return False

    def _update_velocity(self, center):
        """Velocidade (px/quadro) da bola, suavizada, para prever onde ela
        está nos quadros em que a rede não a vê."""
        if self.last_center is None:
            return
        steps = self.missed_frames + 1
        vx = (center[0] - self.last_center[0]) / steps
        vy = (center[1] - self.last_center[1]) / steps
        self.velocity = (0.5 * self.velocity[0] + 0.5 * vx, 0.5 * self.velocity[1] + 0.5 * vy)

    def _rescue_by_color(self, frame, player_boxes=None):
        """Procura a bola por cor perto da posição prevista. Só age logo
        após uma detecção confirmada (nunca inicia uma trajetória) e por
        poucos quadros seguidos."""
        if (
            self.last_center is None
            or self.missed_frames >= self.rescue_window
            or self.rescue_streak >= self.max_rescue_streak
        ):
            return None

        height, width = frame.shape[:2]
        scale = width / 1920.0
        steps = self.missed_frames + 1
        px = self.last_center[0] + self.velocity[0] * steps
        py = self.last_center[1] + self.velocity[1] * steps
        radius = int((30 + 22 * self.missed_frames) * scale) + 6
        x0, x1 = max(0, int(px) - radius), min(width, int(px) + radius)
        y0, y1 = max(0, int(py) - radius), min(height, int(py) + radius)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None

        hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        # Filtro de cor propositalmente frouxo (a bola minúscula perde
        # saturação ao se mover): o saibro fica de fora pelo matiz (H < 17)
        # e o que é amarelo mas não é bola cai nos filtros abaixo, já que a
        # saturação NÃO separa bola de raquete amarelo-esverdeada: raquete
        # (caixa do jogador) e capim seco (sem movimento entre quadros).
        mask = cv2.inRange(hsv, (17, self.RESCUE_MIN_SATURATION, 120), (30, 255, 255))
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        )
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

        # Diferença entre este quadro e o anterior, só na região de busca
        motion = None
        if len(self.frames) >= 2:
            previous = self.frames[-2][y0:y1, x0:x1]
            motion = cv2.absdiff(
                cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY),
            )

        min_area, max_area = 5 * scale * scale, 500 * scale * scale
        best, best_dist = None, None
        for k in range(1, count):
            w = stats[k, cv2.CC_STAT_WIDTH]
            h = stats[k, cv2.CC_STAT_HEIGHT]
            area = stats[k, cv2.CC_STAT_AREA]
            if not min_area <= area <= max_area or max(w, h) > 2.6 * max(1, min(w, h)):
                continue
            if np.median(hsv[:, :, 1][labels == k]) < self.RESCUE_MIN_BLOB_SATURATION:
                continue
            cx, cy = centroids[k][0] + x0, centroids[k][1] + y0
            if self._in_player_zone((cx, cy), player_boxes, scale):
                continue
            if motion is not None:
                r = int(np.sqrt(area / np.pi)) + 2
                mx, my = int(centroids[k][0]), int(centroids[k][1])
                window = motion[max(0, my - r): my + r + 1, max(0, mx - r): mx + r + 1]
                if window.mean() < self.RESCUE_MIN_MOTION:
                    continue
            dist = np.hypot(cx - px, cy - py)
            if dist <= radius and (best is None or dist < best_dist):
                best, best_dist = (int(round(cx)), int(round(cy))), dist
        return best

    @staticmethod
    def _to_world(point, homography):
        if homography is None:
            return None
        try:
            return homography.transform_point_to_world(point)
        except Exception:
            return None

    def _fill_gap(self, center, homography, max_jump):
        """Se a bola reaparece depois de poucos frames sumida e perto de
        onde estava, preenche o rastro (pixels e mundo) desses frames por
        interpolação linear. Não atrasa a saída: só o histórico muda."""
        gap = self.missed_frames
        if self.last_center is None or not 0 < gap <= self.max_gap:
            return
        if gap > len(self.trajectory_pixels) - 1:
            return
        x0, y0 = self.last_center
        if np.hypot(center[0] - x0, center[1] - y0) > max_jump * (gap + 1):
            return
        # trajectory_pixels[0] é o frame anterior; [gap] é a última detecção
        for i in range(gap):
            t = (gap - i) / (gap + 1.0)
            point = (
                int(round(x0 + (center[0] - x0) * t)),
                int(round(y0 + (center[1] - y0) * t)),
            )
            self.trajectory_pixels[i] = point
            self.trajectory_world[i] = self._to_world(point, homography)

    def draw_ball_trail(self, frame):
        """Desenha a linha amarela de trajetória e o marcador da bola."""
        points = list(self.trajectory_pixels)
        for p1, p2 in zip(points, points[1:]):
            if p1 is not None and p2 is not None:
                cv2.line(frame, p1, p2, (0, 255, 255), 2, cv2.LINE_AA)

        if points and points[0] is not None:
            cv2.circle(frame, points[0], 6, (0, 0, 255), 2)
            cv2.circle(frame, points[0], 2, (0, 255, 255), -1)
        return frame
