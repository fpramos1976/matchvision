# MatchVision

## Como rodar

```
uv run python -m app.main                          # vídeo padrão (videos/tennis_3.mp4)
uv run python -m app.main videos/tennis_1.mp4      # outro vídeo
uv run python -m app.main videos/tennis_1.mp4 --classico  # bola pelo método clássico
uv run python -m app.main --help                   # todas as opções
```

A bola é detectada pelo TrackNet (`models/tracknet.pt`) por padrão. Se os
pesos não estiverem disponíveis, o programa usa o detector clássico.
