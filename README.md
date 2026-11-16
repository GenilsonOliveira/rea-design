# rea-design

Uma skill e ferramentas locais para ajudar agentes a investigar a construção visual de imagens, posts e banners e produzir um plano de recriação editável. Inspirado nas investigações orientadas por evidências do [REA](https://github.com/morluto/rea) e nos relatórios estruturados do [rea-video](https://github.com/JuneYaooo/rea-video).

**Estado: versão experimental.** A CLI mede e prepara evidências, executa OCR e segmentação semântica local, propõe um mapa de camadas e exporta uma aproximação em SVG. Um agente com visão ou uma pessoa deve revisar os resultados. Uma imagem raster achatada não contém, em geral, as camadas originais: ordem, máscaras, fontes e partes ocultas são hipóteses, não extrações garantidas.

## Instalação

Requer Python 3.10+ e Pillow. Em um clone deste repositório:

```bash
python -m pip install .
# OCR e segmentação local (opcional):
python -m pip install ".[vision]"
```

No PowerShell, entre na pasta que contém `pyproject.toml` antes de instalar e confira a versão com `rea-design --version`. Se o PATH chamar uma instalação anterior, use `python -m rea_design.cli --version` com o mesmo Python usado pelo pip.

Instale a pasta completa `skills/rea-design` no diretório de skills do seu agente, ou use o instalador de skills compatível com seu agente apontando para este repositório. Para instalação manual no Codex, copie-a para `~/.codex/skills/rea-design`; em Claude Code, para `~/.claude/skills/rea-design`. Reinicie o agente se ele não descobrir a skill. Para outros agentes, forneça o conteúdo de `SKILL.md` e sua pasta `references` como instruções e disponibilize a CLI; compatibilidade depende dos recursos do agente.

Exemplo de pedido: “Use rea-design para analisar `referencia.png`, explique as sobreposições e crie uma receita editável.”

```bash
rea-design inspect referencia.png --out analise/evidence
rea-design ocr referencia.png --out analise/ocr
rea-design segment referencia.png --out analise/segments --rect 57 76 290 410
rea-design compare referencia.png render.png --out analise/comparison
```

### Segmentação semântica e SVG

Além de `.[vision]`, baixe separadamente o [FCN-ResNet50-12 do ONNX Model Zoo](https://huggingface.co/onnxmodelzoo/fcn-resnet50-12) (aproximadamente 141 MB) e forneça o caminho com `--model`. O projeto não baixa nem redistribui o modelo. Por exemplo, no PowerShell, dentro da pasta do repositório:

```powershell
New-Item -ItemType Directory -Force models | Out-Null
Invoke-WebRequest 'https://huggingface.co/onnxmodelzoo/fcn-resnet50-12/resolve/f2d2cd1/fcn-resnet50-12.onnx?download=true' -OutFile 'models/fcn-resnet50-12.onnx'
Get-FileHash 'models/fcn-resnet50-12.onnx' -Algorithm SHA256
```

O SHA-256 do modelo usado nos testes é `eb5017d1b80372eb0b58552655274698817ba2e774437e2c2d3c0a613d2e99bd`. Depois, com pastas de saída novas:

```powershell
rea-design ocr referencia.png --out analise/ocr
rea-design semantic referencia.png --model models/fcn-resnet50-12.onnx --out analise/semantic
rea-design scene referencia.png --semantic analise/semantic/semantic.json --ocr analise/ocr/ocr.json --out analise/scene
rea-design export-svg referencia.png --scene analise/scene/scene.json --out analise/reconstrucao.svg
```

O FCN classifica pixels em 20 categorias do Pascal VOC, como pessoa, garrafa e carro; não identifica livremente logos, produtos, formas ou todas as pessoas. Componentes desconectados da mesma classe podem representar partes de um único objeto. OCR e segmentação exigem a mesma imagem e registram seu hash; dados antigos de OCR sem hash precisam ser regenerados. O arquivo `scene.json` relaciona camadas, máscaras e interseções de pixels: uma interseção não prova oclusão e a profundidade é `unknown`. `render_order` é apenas uma ordem inicial e pode ser editada junto com textos e cores antes da exportação.

Para mover uma cena gerada pela 0.2.1, mantenha `scene.json` junto de `scene-masks/`. O exportador valida o hash das máscaras copiadas. Cenas antigas continuam abrindo com um aviso sobre máscaras sem hash. Gere a cena novamente para incluí-las no pacote. Arquivos JSON salvos pelo PowerShell 5.1 em UTF‑8 com BOM também são aceitos.

O SVG é autocontido. Contém texto editável e recortes raster separados sobre um fundo raster reconstruído por inpainting, preservando o canal alfa. A limpeza do texto estima os pixels das letras pela cor. A saída inclui avisos quando a estimativa falha ou quando o fundo precisa ficar transparente por falta de pixels disponíveis. Texto residual, artefatos e partes não detectadas podem permanecer no fundo. Tipografia, máscaras e ordem são estimativas. Veja [os testes com duas imagens reais](experiments/RESULTS.md).

### Corrigir máscaras e reconstruir a partir de uma foto limpa

`scene` aceita OCR, segmentação semântica, recortes GrabCut e máscaras manuais em qualquer combinação. Para importar um recorte assistido e corrigir seu contorno:

```powershell
rea-design segment referencia.png --rect 98 0 538 920 --out analise/recorte
rea-design scene referencia.png --ocr analise/ocr/ocr.json --grabcut analise/recorte/segments.json --out analise/cena-inicial
rea-design scene referencia.png --ocr analise/ocr/ocr.json --grabcut analise/recorte/segments.json --replace-mask grabcut-001 mascara-revisada.png --out analise/cena
rea-design export-svg referencia.png --scene analise/cena/scene.json --clean-plate foto-alinhada.png --out analise/editavel.svg
```

Use `--mask "nome da camada" caminho.png` para adicionar uma camada manual (`manual-001`, `manual-002`, etc.). As opções `--grabcut`, `--mask` e `--replace-mask` podem ser repetidas. As máscaras devem ter o tamanho do canvas. Use tons de cinza ou transparência para definir a cobertura. A correção mantém o ID e recalcula os limites e interseções. Gere novamente `segments.json` se o arquivo antigo tiver sido criado antes da inclusão do hash do original.

`--clean-plate` usa uma foto sem os elementos gráficos, com o mesmo tamanho e alinhamento da cena. Confira o alinhamento visual e ajuste as máscaras ao contorno da foto. Nessas cenas, é possível acrescentar retângulos vetoriais com `id`, `kind: "rect"`, `bbox: [x, y, largura, altura]`, `fill` e `opacity` opcional. Inclua os IDs na posição desejada em `render_order`. Isso permite reconstruir células e linhas de grade como elementos que podem ser movidos ou recoloridos. A posição desses elementos é indicada pelo agente ou pela pessoa que revisa a cena.

Se mover o recorte revelar artefatos do preenchimento automático, passe também `--background-plate fundo-sem-o-sujeito.png`. A imagem deve ter o tamanho do canvas. O SVG usa esse fundo sob os recortes e registra o hash da imagem na resposta da CLI.

### Detectar grades e blocos de cor

```powershell
rea-design geometry referencia.png --out analise/geometria
rea-design scene referencia.png --geometry analise/geometria/geometry.json --out analise/cena
rea-design export-svg referencia.png --scene analise/cena/scene.json --out analise/editavel.svg
```

`geometry` detecta traços horizontais e verticais finos com apoio contínuo e preenche propostas de blocos quando quatro linhas delimitam uma célula de cor uniforme com contraste externo. `geometry-overlay.png` ajuda a revisar as propostas. O detector não busca blocos sem linhas delimitadoras nem grades diagonais ou curvas. A espessura exportada é de 1 px e a opacidade é estimada.

`scene --geometry` pode ser combinado com `--ocr` e as opções de recorte. O SVG exporta cada forma como um `<rect>` separado. Edite suas coordenadas ou `fill` no editor SVG. Ao editar `scene.json`, conserve `source_bbox` para indicar a posição original da forma e altere `bbox` para movê-la. Sem `--clean-plate`, as regiões originais detectadas passam por preenchimento estimado. Para revelar uma foto coberta por um bloco, use uma foto limpa e alinhada quando disponível.

`inspect` gera dimensões, hash, paleta aproximada, imagem normalizada, grade, mapa de bordas e quatro recortes com contexto. A paleta é quantizada sobre branco. `compare` exige imagens do mesmo tamanho e mede separadamente o erro RGB sobre branco e o erro no canal alfa. Produz também `alpha-difference.png`. A CLI bloqueia caminhos de saída que sobrescrevam suas entradas. Outros arquivos de saída existentes podem ser substituídos.

`ocr` usa RapidOCR/ONNX localmente e gera texto, polígonos, confiança e uma imagem de verificação. Confiança alta não garante transcrição correta; confira letras decorativas e idiomas. `segment` produz máscaras de componentes de cor e caixas em coordenadas da imagem. São regiões visuais, não classes semânticas nem camadas originais. Opcionalmente, `--rect X Y LARGURA ALTURA` seleciona um retângulo aproximado em pixels para separar o primeiro plano via GrabCut, salvando uma máscara e um recorte transparente. O retângulo deve incluir o sujeito e deixar fundo fora dele; o resultado precisa de inspeção. `--colors` controla a quantização (2–16) e `--min-area` o limiar de área como fração da imagem (padrão 0,001). Saídas anteriores em uma pasta reutilizada podem permanecer; use sempre uma pasta nova.

Veja o [experimento com anúncio histórico](examples/vintage-ad/DESIGN.md), incluindo transcrição corrigida, máscaras, recriação e diferenças. Ele é um único caso, não uma medição geral de qualidade.

O agente usa as evidências para criar `DESIGN.md`: composição, tipografia, mapa de camadas e oclusões, receita de reconstrução e incertezas. Esse documento é gerado pelo agente, não automaticamente pela CLI. Para reconstruir, é necessário também um editor ou ferramenta de renderização disponível ao agente. As imagens são processadas localmente pela CLI; o envio de imagens a um modelo depende do seu agente e provedor.

## Desenvolvimento

```bash
python -m pip install -e .
python -m unittest discover -s tests
```

## Limites e próximos passos

Esta versão aceita como entrada formatos raster que o Pillow consegue abrir. Entrada SVG, PSD ou PDF, processamento em lote, segmentação semântica de vocabulário aberto, detecção de fontes e exportação para PSD não estão implementados. Os resultados variam conforme imagem, idioma e modelos; dois exemplos não estabelecem fidelidade. Use materiais com direitos adequados ao contexto.
