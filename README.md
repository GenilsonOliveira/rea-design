# rea-design

Uma skill e ferramentas locais para ajudar agentes a investigar a construção visual de imagens, posts e banners e produzir um plano de recriação editável. Inspirado nas investigações orientadas por evidências do [REA](https://github.com/morluto/rea) e nos relatórios estruturados do [rea-video](https://github.com/JuneYaooo/rea-video).

**Estado: versão experimental.** A CLI mede e prepara evidências, executa OCR local e segmenta regiões; um agente com visão interpreta os resultados e escreve o relatório. Uma imagem raster achatada não contém, em geral, as camadas originais: ordem, máscaras, fontes e partes ocultas são hipóteses, não extrações garantidas.

## Instalação

Requer Python 3.10+ e Pillow. Em um clone deste repositório:

```bash
python -m pip install .
# OCR e segmentação local (opcional):
python -m pip install ".[vision]"
```

Instale a pasta completa `skills/rea-design` no diretório de skills do seu agente, ou use o instalador de skills compatível com seu agente apontando para este repositório. Para instalação manual no Codex, copie-a para `~/.codex/skills/rea-design`; em Claude Code, para `~/.claude/skills/rea-design`. Reinicie o agente se ele não descobrir a skill. Para outros agentes, forneça o conteúdo de `SKILL.md` e sua pasta `references` como instruções e disponibilize a CLI; compatibilidade depende dos recursos do agente.

Exemplo de pedido: “Use rea-design para analisar `referencia.png`, explique as sobreposições e crie uma receita editável.”

```bash
rea-design inspect referencia.png --out analise/evidence
rea-design ocr referencia.png --out analise/ocr
rea-design segment referencia.png --out analise/segments --rect 57 76 290 410
rea-design compare referencia.png render.png --out analise/comparison
```

`inspect` gera dimensões, hash, paleta aproximada, imagem normalizada, grade, mapa de bordas e quatro recortes com contexto. As cores são quantizadas após composição sobre branco; não são tokens originais. `compare` exige imagens do mesmo tamanho e gera sobreposição, diferença e erro médio por canal. Arquivos de mesmo nome no diretório de saída são substituídos: use uma pasta nova por execução.

`ocr` usa RapidOCR/ONNX localmente e gera texto, polígonos, confiança e uma imagem de verificação. Confiança alta não garante transcrição correta; confira letras decorativas e idiomas. `segment` produz máscaras de componentes de cor e caixas em coordenadas da imagem. São regiões visuais, não classes semânticas nem camadas originais. Opcionalmente, `--rect X Y LARGURA ALTURA` seleciona um retângulo aproximado em pixels para separar o primeiro plano via GrabCut, salvando uma máscara e um recorte transparente. O retângulo deve incluir o sujeito e deixar fundo fora dele; o resultado precisa de inspeção. `--colors` controla a quantização (2–16) e `--min-area` o limiar de área como fração da imagem (padrão 0,001). Saídas anteriores em uma pasta reutilizada podem permanecer; use sempre uma pasta nova.

Veja o [experimento com anúncio histórico](examples/vintage-ad/DESIGN.md), incluindo transcrição corrigida, máscaras, recriação e diferenças. Ele é um único caso, não uma medição geral de qualidade.

O agente usa as evidências para criar `DESIGN.md`: composição, tipografia, mapa de camadas e oclusões, receita de reconstrução e incertezas. Esse documento é gerado pelo agente, não automaticamente pela CLI. Para reconstruir, é necessário também um editor ou ferramenta de renderização disponível ao agente. As imagens são processadas localmente pela CLI; o envio de imagens a um modelo depende do seu agente e provedor.

## Desenvolvimento

```bash
python -m pip install -e .
python -m unittest discover -s tests
```

## Limites e próximos passos

Esta versão aceita formatos raster que o Pillow consegue abrir. SVG, PSD, PDF, processamento em lote, segmentação semântica automática, detecção de fontes e exportação em camadas não estão implementados. Os resultados variam conforme imagem, idioma e modelos; um único exemplo não estabelece fidelidade. Use materiais com direitos adequados ao contexto.
