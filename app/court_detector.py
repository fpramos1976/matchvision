import cv2
import numpy as np


def detect_court_lines(frame):
    """Detecta e filtra as linhas da quadra focando apenas na área útil do vídeo."""
    if frame is None:
        return None, None

    output_frame = frame.copy()
    height, width = frame.shape[:2]

    # 1. Criar uma máscara simples para cortar o céu e o topo das árvores
    # Vamos focar apenas do meio da tela para baixo (a partir de 42% da altura)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    # Pinta de branco apenas a região onde a quadra realmente pode estar
    cv2.rectangle(
        mask, (0, int(height * 0.42)), (width, int(height * 0.95)), 255, -1
    )

    # 2. Pré-processamento apenas na área permitida pela máscara
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_masked = cv2.bitwise_and(gray, gray, mask=mask)
    blurred = cv2.GaussianBlur(gray_masked, (5, 5), 0)

    # 3. Canny - Ajustado para destacar melhor as linhas brancas no saibro
    edges = cv2.Canny(blurred, 70, 180)

    # 4. Hough Lines com parâmetros refinados para este ângulo
    # Reduzimos o maxLineGap para evitar que ruídos distantes se conectem sozinhos
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=45,
        minLineLength=40,
        maxLineGap=15,
    )

    # 5. Filtragem Geométrica Restrita
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.ravel()

            # Calcula comprimento e ângulo
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            angle = np.abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

            # Filtro de inclinação rígido para esta perspectiva
            is_horizontal = angle < 5 or angle > 175
            is_diagonal_court = (20 < angle < 65) or (115 < angle < 160)

            if is_horizontal:
                # Desenha linhas horizontais legítimas em AZUL
                cv2.line(output_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            elif is_diagonal_court:
                # Evita linhas excessivamente verticais ou gigantes que passam do meio da quadra
                if length < height * 0.5:
                    cv2.line(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return output_frame, edges