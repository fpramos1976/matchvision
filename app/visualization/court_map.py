import cv2
import numpy as np


class CourtMap:

    def __init__(self, court_model, pixels_per_meter=25, margin=40):
        """Gera a representação 2D (visão superior) da quadra.

        :param court_model: Objeto CourtModel com largura e comprimento reais
        :param pixels_per_meter: Escala de conversão metros -> pixels
        :param margin: Margem externa ao redor da quadra em pixels
        """
        self.court_model = court_model
        self.ppm = pixels_per_meter
        self.margin = margin

        # Dimensões da imagem 2D
        self.width = int(court_model.width * self.ppm) + (margin * 2)
        self.height = int(court_model.length * self.ppm) + (margin * 2)

    def _meters_to_pixels(self, x_m, y_m):
        """Converte coordenadas em metros para pixels no mapa 2D."""
        px = int(x_m * self.ppm) + self.margin
        py = int(y_m * self.ppm) + self.margin
        return (px, py)

    def create(self):
        """Método legado de compatibilidade (retorna a quadra base)."""
        return self.draw_base_court()

    def draw_base_court(self):
        """Desenha a estrutura base da quadra de tênis."""
        # Fundo verde-escuro
        map_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        map_img[:] = (34, 102, 34)  # Verde escuro BGR

        # 1. Perímetro Externo
        p1 = self._meters_to_pixels(0, 0)
        p2 = self._meters_to_pixels(
            self.court_model.width, self.court_model.length
        )
        cv2.rectangle(map_img, p1, p2, (255, 255, 255), 2)

        # 2. Rede
        net_y = self.court_model.length / 2
        np1 = self._meters_to_pixels(-0.5, net_y)
        np2 = self._meters_to_pixels(self.court_model.width + 0.5, net_y)
        cv2.line(map_img, np1, np2, (200, 200, 200), 3)

        # 3. Linhas de Saque (6.4m da rede)
        service_line_dist = 6.4
        y_service_top = net_y - service_line_dist
        y_service_bottom = net_y + service_line_dist

        sp1 = self._meters_to_pixels(0, y_service_top)
        sp2 = self._meters_to_pixels(self.court_model.width, y_service_top)
        cv2.line(map_img, sp1, sp2, (255, 255, 255), 2)

        sp3 = self._meters_to_pixels(0, y_service_bottom)
        sp4 = self._meters_to_pixels(self.court_model.width, y_service_bottom)
        cv2.line(map_img, sp3, sp4, (255, 255, 255), 2)

        # 4. Linha Central de Saque (T)
        center_x = self.court_model.width / 2
        cp1 = self._meters_to_pixels(center_x, y_service_top)
        cp2 = self._meters_to_pixels(center_x, y_service_bottom)
        cv2.line(map_img, cp1, cp2, (255, 255, 255), 2)

        return map_img

    def draw_players(self, map_img, players):
        """Plota os jogadores e o painel de métricas no Bird's-Eye View."""
        for player in players:
            p_id = player["id"]
            world_x, world_y = player["world_pos"]
            px, py = self._meters_to_pixels(world_x, world_y)

            if 0 <= px < self.width and 0 <= py < self.height:
                # Círculo do Jogador
                cv2.circle(map_img, (px, py), 10, (0, 0, 0), -1)
                cv2.circle(map_img, (px, py), 8, (0, 255, 255), -1)

                # Número/ID do jogador no centro do círculo
                cv2.putText(
                    map_img,
                    str(p_id),
                    (px - 4, py + 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 0),
                    1,
                )

        # Draw HUD / Painel de Estatísticas
        self._draw_stats_hud(map_img, players)

        return map_img

    def _draw_stats_hud(self, map_img, players):
        """Desenha um painel transparente com as estatísticas dos jogadores."""
        hud_height = 70
        overlay = map_img.copy()
        cv2.rectangle(
            overlay, (0, 0), (self.width, hud_height), (20, 20, 20), -1
        )
        # Aplica transparência no HUD
        cv2.addWeighted(overlay, 0.7, map_img, 0.3, 0, map_img)

        y_offset = 20
        for p in players:
            text = f"P{p['id']}: {p['total_distance']:.1f}m | {p['speed']:.1f} km/h"
            cv2.putText(
                map_img,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )
            y_offset += 22

    def draw_ball(self, map_img, ball_world_pos):
        """Plota a bolinha no Bird's-Eye View."""
        if ball_world_pos:
            wx, wy = ball_world_pos
            px, py = self._meters_to_pixels(wx, wy)

            if 0 <= px < self.width and 0 <= py < self.height:
                # Desenha ponto amarelo chamativo da bola
                cv2.circle(map_img, (px, py), 5, (0, 0, 0), -1)
                cv2.circle(map_img, (px, py), 4, (0, 255, 255), -1)

        return map_img