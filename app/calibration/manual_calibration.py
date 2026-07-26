import cv2


class ManualCalibration:
    def __init__(self, frame, configuration):
        self.frame = frame.copy()
        self.display_frame = frame.copy()

        self.configuration = configuration

        self.points = []

        self.window_name = "Manual Calibration"

        self.instructions = [
            "Clique no canto SUPERIOR ESQUERDO",
            "Clique no canto SUPERIOR DIREITO",
            "Clique no canto INFERIOR DIREITO",
            "Clique no canto INFERIOR ESQUERDO",
        ]

    def show_instruction(self):
        """Mostra a próxima instrução ao usuário."""

        index = len(self.points)

        if index < len(self.instructions):
            print()
            print("=" * 50)
            print(self.instructions[index])
            print("=" * 50)

    def mouse_callback(self, event, x, y, flags, param):
        """Captura os cliques do mouse."""

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if len(self.points) >= 4:
            return

        # Salva o ponto
        self.points.append((x, y))

        print(f"Ponto {len(self.points)}: ({x}, {y})")

        # Desenha um círculo vermelho
        cv2.circle(self.display_frame, (x, y), 6, (0, 0, 255), -1)

        # Liga o ponto atual ao anterior
        if len(self.points) > 1:
            cv2.line(
                self.display_frame,
                self.points[-2],
                self.points[-1],
                (0, 255, 0),
                2,
            )

        # Fecha o polígono quando houver 4 pontos
        if len(self.points) == 4:
            cv2.line(
                self.display_frame,
                self.points[3],
                self.points[0],
                (0, 255, 0),
                2,
            )

        cv2.imshow(self.window_name, self.display_frame)

        # Mostra a próxima instrução
        self.show_instruction()

    def run(self):
        """Executa a calibração manual."""

        print("\n=== Calibração Manual ===")
        print(f"Tipo de partida: {self.configuration.court_type.value}")
        print("Pressione Q para cancelar.\n")

        self.show_instruction()

        cv2.namedWindow(self.window_name)

        cv2.setMouseCallback(
            self.window_name,
            self.mouse_callback,
        )

        while True:
            cv2.imshow(self.window_name, self.display_frame)

            if len(self.points) == 4:
                print("\nCalibração concluída!\n")

                for i, point in enumerate(self.points, start=1):
                    print(f"P{i}: {point}")

                break

            key = cv2.waitKey(20) & 0xFF

            if key == ord("q"):
                print("Calibração cancelada.")
                break

        cv2.destroyWindow(self.window_name)

        return self.points