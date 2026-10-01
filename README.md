# MatchVision

## Como rodar

```
uv run python -m app.main                          # vídeo padrão (videos/tennis_3.mp4)
uv run python -m app.main videos/tennis_1.mp4      # outro vídeo
uv run python -m app.main videos/tennis_1.mp4 --classico  # bola pelo método clássico
uv run python -m app.main --help                   # todas as opções
```

O vídeo é processado primeiro (o terminal mostra o progresso) e o resultado
anotado é gravado em `output/<nome>_analise.mp4`. No fim, ele é reproduzido
na velocidade normal: espaço pausa, R reinicia, Q sai. Use `--sem-reproducao`
para só gravar, e `--saida arquivo.mp4` para escolher onde salvar.

A bola é detectada pelo TrackNet (`models/tracknet.pt`) por padrão. Se os
pesos não estiverem disponíveis, o programa usa o detector clássico.
