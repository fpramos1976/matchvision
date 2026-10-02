---
name: ponytail
description: Revisão completa das mudanças da branch atual — roda /code-review, /security-review e /simplify em sequência e entrega um resumo consolidado. Use quando o usuário digitar /ponytail ou pedir uma "revisão completa" das mudanças.
---

# /ponytail — revisão completa

Revise as mudanças pendentes da branch atual (diff contra a branch padrão, mais alterações não commitadas) executando, nesta ordem, as skills abaixo com a ferramenta Skill. Repasse para `/code-review` qualquer argumento que o usuário tenha dado ao `/ponytail` (nível de esforço, número de PR, `--comment`, etc.).

1. **`code-review`** — bugs de corretude. Anote os achados; não aplique correções ainda.
2. **`security-review`** — vulnerabilidades e riscos de segurança nas mudanças. Anote os achados.
3. **`simplify`** — limpezas de reuso, simplificação e eficiência. Esta etapa aplica as correções no working tree.

Se não houver mudanças para revisar, diga isso e pare.

## Resumo final

Depois das três etapas, responda em português com um resumo único:

- **Bugs** (do code-review), do mais grave para o menos grave, com `arquivo:linha`.
- **Segurança** (do security-review), com severidade e `arquivo:linha`.
- **Simplificações aplicadas** (do simplify): o que mudou e em quais arquivos.
- **Próximos passos**: o que ainda precisa de decisão do usuário.

Não faça commit nem push das alterações do simplify sem o usuário pedir.
