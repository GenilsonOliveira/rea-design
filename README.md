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

### Reconstruir um título editável

```powershell
rea-design ocr referencia.png --out analise/ocr
rea-design typography referencia.png --ocr analise/ocr/ocr.json --out analise/tipografia
rea-design scene referencia.png --ocr analise/ocr/ocr.json --typography analise/tipografia/typography.json --out analise/cena
rea-design export-svg referencia.png --scene analise/cena/scene.json --out analise/editavel.svg
```

`typography` propõe o título dominante. Agrupa trechos de OCR alinhados e próximos, mede a extensão dos pixels das letras e repete a leitura nas linhas separadas. O relatório inclui caixa, cor, confiança e cobertura de tinta por linha, além de uma sugestão de fonte. Títulos muito decorativos, inclinados ou com contraste baixo precisam de revisão. Veja `typography-overlay.png` para conferir as caixas.

O título aparece como um grupo `text-block` com uma tag SVG `<text>` por linha. Para ajustar a frase ao mesmo espaço, edite `lines[].text` no `scene.json` e exporte novamente. Em renderizadores com suporte a `textLength`, é possível editar o texto diretamente no SVG. Use `export-svg --title-font caminho.ttf` para medir as letras do texto final com um arquivo local e calcular também a transformação geométrica. O editor deve ter acesso à família usada. O SVG guarda texto editável e referencia a fonte pelo nome. Opcionalmente, `--clean-plate foto-alinhada.png` preserva a foto de base ao trocar o título.

Quando as caixas de OCR de linhas vizinhas se cruzam, a análise limita os recortes para medir cada linha localmente. A leitura inicial é preservada quando tem boa confiança e a releitura discorda. Linhas de um título com cor de tinta consistente podem ajudar a localizar a mesma cor sobre fotografia ou papel texturizado. Revise textos manuscritos, cores misturadas em uma linha e letras encobertas.

Para substituir apenas o título a partir da arte achatada, use a cena com as máscaras dos objetos:

```powershell
rea-design export-svg poster.png --scene analise/scene/scene.json --title-only --title-line "MY NEXT" --title-line "CHAPTER" --title-font caminho.ttf --out editado.svg
```

`--title-only` recorta e preenche os pixels visíveis das letras originais. O exportador tenta identificar onde a tinta termina junto ao contorno de um objeto e mantém esse recorte à frente do título. O resultado mantém a composição restante como imagem raster e o título como texto SVG. Para uma oclusão ambígua, use `--title-behind <layer-id>` para indicar o objeto que fica sobre o título. Repita a opção para informar mais de um. Revise o contorno do objeto e as áreas preenchidas no SVG.

Para destacar uma palavra manuscrita e mantê-la acima do título novo, adicione `--title-overlay text-005 "Building"` ao comando. Escolha em `scene.json` o ID do texto detectado pelo OCR. A caixa de OCR pode cobrir só parte da palavra. O rea-design procura os traços conectados, separa a tinta do papel e do título antigo e exporta a palavra como uma imagem transparente em uma camada própria. O rótulo dá nome à camada. `--move text-005 24 -48` desloca apenas essa camada. A separação depende do contraste entre as tintas e o fundo. Revise as bordas e a região revelada quando mover o recorte.

Ao limpar um manuscrito, o rea-design estima o tom local do papel e reaproveita o grão de áreas visíveis do próprio pôster. A correção fica restrita à máscara de limpeza. Onde faltam amostras de papel compatíveis, vale o preenchimento clássico. Confira a chave `paper_texture_pixels` no relatório da exportação para saber quantos pixels passaram pelo reparo de textura.

`inspect` gera dimensões, hash, paleta aproximada, imagem normalizada, grade, mapa de bordas e quatro recortes com contexto. A paleta é quantizada sobre branco. `compare` exige imagens do mesmo tamanho e mede separadamente o erro RGB sobre branco e o erro no canal alfa. Produz também `alpha-difference.png`. A CLI bloqueia caminhos de saída que sobrescrevam suas entradas. Outros arquivos de saída existentes podem ser substituídos.

`ocr` usa RapidOCR/ONNX localmente e gera texto, polígonos, confiança e uma imagem de verificação. Confiança alta não garante transcrição correta; confira letras decorativas e idiomas. `segment` produz máscaras de componentes de cor e caixas em coordenadas da imagem. São regiões visuais, não classes semânticas nem camadas originais. Opcionalmente, `--rect X Y LARGURA ALTURA` seleciona um retângulo aproximado em pixels para separar o primeiro plano via GrabCut, salvando uma máscara e um recorte transparente. O retângulo deve incluir o sujeito e deixar fundo fora dele; o resultado precisa de inspeção. `--colors` controla a quantização (2–16) e `--min-area` o limiar de área como fração da imagem (padrão 0,001). Saídas anteriores em uma pasta reutilizada podem permanecer; use sempre uma pasta nova.

Veja o [experimento com anúncio histórico](examples/vintage-ad/DESIGN.md), incluindo transcrição corrigida, máscaras, recriação e diferenças. Ele é um único caso, não uma medição geral de qualidade.

O agente usa as evidências para criar `DESIGN.md`: composição, tipografia, mapa de camadas e oclusões, receita de reconstrução e incertezas. Esse documento é gerado pelo agente, não automaticamente pela CLI. Para reconstruir, é necessário também um editor ou ferramenta de renderização disponível ao agente. As imagens são processadas localmente pela CLI; o envio de imagens a um modelo depende do seu agente e provedor.

### Recortar um elemento pela descrição

Instale `.[vision,prompt]` no ambiente Python que vai executar a CLI. A inferência roda em CPU, com a imagem no seu computador. Baixe os pesos uma vez, na pasta do projeto. As revisões abaixo foram usadas no teste e baixam cerca de 1,1 GB em arquivos de modelo.

```powershell
python -m pip install ".[vision,prompt]"
python -m rea_design.cli fetch-prompt-models --models-dir models
```

```powershell
rea-design segment-text referencia.png --prompt "recorte só os óculos de sol" --models-dir models --out analise/oculos
rea-design scene referencia.png --grounded analise/oculos/grounded.json --out analise/cena
rea-design export-svg referencia.png --scene analise/cena/scene.json --out analise/editavel.svg
```

O Grounding DINO identifica a região candidata pelo texto. O Segment Anything recorta os pixels visíveis. Prefira o nome curto do objeto em inglês. Algumas expressões em português, como “óculos de sol”, têm aliases locais. O comando retorna uma correspondência por execução com overlay, máscara do tamanho do canvas e recorte transparente. `grounded.json` indica `no-match` se nenhum resultado passar pelo limiar. Confira a máscara antes de importá-la. `scene --grounded` pode ser combinado com `--ocr`, `--geometry` e as outras fontes da cena. O SVG mantém o recorte em um grupo raster que pode ser movido e redimensionado. Para um SVG apenas com o recorte, forneça uma imagem transparente do tamanho do canvas em `export-svg --background-plate`.

### Montar um SVG com vários recortes

```powershell
rea-design assemble referencia.png --prompt "óculos de sol" --prompt "pessoa" --models-dir models --clean-plate retrato-com-oculos.png --repair-model models/lama/lama_fp32.onnx --title-font caminho.ttf --out analise
rea-design export-svg referencia.png --scene analise/scene/scene.json --clean-plate retrato-com-oculos.png --repair-model models/lama/lama_fp32.onnx --title-font caminho.ttf --move grounded-001 32 -120 --title-line NOVA --title-line FORMA --out analise/editado.svg
```

`assemble` mede a referência, procura um recorte por descrição em uma execução e monta `reconstruction.svg`. O `scene.json` registra cada interseção. Quando uma máscara menor está quase toda contida em uma maior, a CLI sugere a menor na frente. Interseções ambíguas mantêm profundidade desconhecida. O SVG agrupa as linhas da grade e o texto pequeno à direita do título, quando detectado. O arquivo de cena permite revisar a proposta e ajustar `render_order`.

A foto opcional de `--clean-plate` fornece os pixels dos recortes e deve estar alinhada à referência. Ela mantém os objetos visíveis e pode vir sem a tipografia e a grade. Por padrão, a própria referência fornece os pixels. A segmentação retorna `partial` quando somente algumas descrições têm correspondência.

Para `--repair-model`, instale o extra `vision` e salve o [checkpoint LaMa de 512 × 512](https://huggingface.co/Carve/LaMa-ONNX/blob/main/lama_fp32.onnx) como `models/lama/lama_fp32.onnx`. O modelo roda em CPU e preenche as interseções entre objetos cuja ordem é sugerida ou confirmada e respeitada no `render_order`. A exportação registra as camadas preenchidas, o número de pixels ocultos e o hash do modelo. Revise a área revelada, especialmente se o objeto oculta olhos, letras ou detalhes finos.

Quando houver uma foto alinhada que mostre a área oculta, use `--repair-plate` no lugar de `--repair-model`. Sem essas opções, a CLI usa preenchimento clássico e sinaliza a estimativa. Para controlar o fundo do canvas, use a opção independente `--background-plate`.

### Preenchimento generativo de detalhes ocultos

```powershell
rea-design prepare-repair referencia.png --scene analise/scene/scene.json --clean-plate retrato-com-oculos.png --out analise/reparo
# Use context.png e mask.png no seu gerador de imagens e salve o resultado em analise/reparo/gerado.png.
rea-design export-svg referencia.png --scene analise/scene/scene.json --clean-plate retrato-com-oculos.png --generative-repair analise/reparo/repair.json analise/reparo/gerado.png --move grounded-001 32 -120 --title-line NOVA --title-line FORMA --out analise/editado.svg
```

`prepare-repair` usa as interseções ordenadas da cena para escolher o recorte e criar a máscara. Branco marca a região editável. Use `--layer` para escolher a camada que recebe o preenchimento quando houver mais de uma. `--margin` ajusta a largura da transição em pixels da imagem original, por exemplo para incluir a sombra projetada pelo objeto.

Peça ao gerador para reconstruir o detalhe dentro da máscara e devolver o recorte inteiro, opaco, com o mesmo enquadramento e proporção. Pode usar a ferramenta de imagens do agente ou um modelo local compatível. O rea-design prepara, alinha e compõe o recorte. A geração de pixels fica a cargo da ferramenta escolhida. Na exportação, o rea-design confere a fonte e o recorte pela cena, tenta corrigir pequenos deslocamentos pelo contexto e aplica os pixels dentro da máscara com uma transição suave. Repita `--generative-repair` para diferentes camadas da mesma cena. Os detalhes reconstruídos são uma síntese visual.

## Desenvolvimento

```bash
python -m pip install -e .
python -m unittest discover -s tests
```

## Limites e próximos passos

Esta versão aceita como entrada formatos raster que o Pillow consegue abrir. Entrada SVG, PSD ou PDF, processamento em lote, segmentação semântica de vocabulário aberto, detecção de fontes e exportação para PSD não estão implementados. Os resultados variam conforme imagem, idioma e modelos; dois exemplos não estabelecem fidelidade. Use materiais com direitos adequados ao contexto.
