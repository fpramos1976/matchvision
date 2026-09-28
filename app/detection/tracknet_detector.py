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

    def __init__(
        self,
        model_path="models/tracknet.pt",
        fps=30,
        max_buffer=30,
        heatmap_threshold=127,
        max_jump=100,
        device=None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _TrackNet()
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

        # A rede foi treinada a 30 fps. Em vídeos de 60 fps a bola anda
        # metade por frame e a rede a confunde com algo parado; por isso
        # usamos frames espaçados (t, t-2, t-4) para reproduzir o
        # deslocamento que ela viu no treino.
        self.frame_step = max(1, int(round(fps / 30.0)))
        self.frames = collections.deque(maxlen=2 * self.frame_step + 1)

        self.heatmap_threshold = heatmap_threshold
        # Saltos maiores que isto (em pixels do frame) entre detecções
        # seguidas são tratados como falso positivo.
        self.max_jump = max_jump

        self.trajectory_pixels = collections.deque(maxlen=max_buffer)
        self.trajectory_world = collections.deque(maxlen=max_buffer)
        self.last_center = None
        self.missed_frames = 0

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

        if center is not None and self.last_center is not None and self.missed_frames < 4:
            jump = np.hypot(center[0] - self.last_center[0], center[1] - self.last_center[1])
            if jump > self.max_jump * (1 + self.missed_frames):
                center = None

        world_pos = None
        if center is not None:
            if homography is not None:
                try:
                    world_pos = homography.transform_point_to_world(center)
                except Exception:
                    world_pos = None
            self.last_center = center
            self.missed_frames = 0
        else:
            self.missed_frames += 1

        self.trajectory_pixels.appendleft(center)
        self.trajectory_world.appendleft(world_pos)
        return center, world_pos

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
