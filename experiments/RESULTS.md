# Testes de segmentação e SVG (FCN-ResNet50-12)

Dois testes exploratórios com obras históricas. O modelo e as ferramentas foram executados localmente. O modelo do [ONNX Model Zoo](https://huggingface.co/onnxmodelzoo/fcn-resnet50-12) tinha SHA-256 `eb5017d1b80372eb0b58552655274698817ba2e774437e2c2d3c0a613d2e99bd`. O texto no SVG é OCR bruto: não houve correção manual. Renderização de teste: CairoSVG 2.9.1; outros renderizadores podem apresentar tipografia diferente.

| Referência | Saída semântica | OCR e interseções | Resultado |
| --- | --- | --- | --- |
| [Inca Kola](https://commons.wikimedia.org/wiki/File:Inca_Kola_Advertisement.png), 415×568 | 2 componentes `person`; um é parte do tecido, não uma segunda pessoa | 6 trechos; nenhuma interseção das máscaras de texto e objetos | [SVG](vintage/reconstruction.svg), [prévia](vintage/preview.png), [mapa](vintage/scene/scene.json) |
| [Coca-Cola 1924](https://commons.wikimedia.org/wiki/File:Coca-Cola_ad_1924-04.png), 1500×1700 | 3 componentes `person`, fragmentos da mesma figura | 16 trechos; 6 pares com polígonos intersectados, sem conclusão sobre ordem | [SVG](coca/reconstruction.svg), [prévia](coca/preview.png), [mapa](coca/scene/scene.json) |

As páginas de origem descrevem o estado de domínio público das imagens; as marcas aparecem no contexto das obras históricas. Os arquivos fonte estão em `examples/vintage-ad/reference.png` e `examples/coca-cola-1924/reference.png`. Cada diretório contém JSON de OCR, mapa semântico, máscaras, sobreposições e comparação. No segundo caso, várias interseções são entre linhas de OCR ou um logotipo e um texto pequeno; isso não constitui prova de sobreposição real.

No primeiro caso, o OCR errou “sabor” e o SVG troca a caligrafia por uma fonte genérica. No segundo, o modelo fragmentou a figura, não separou o copo como objeto e o logotipo manuscrito ficou ruim em texto genérico. Ambas as reconstruções incorporam pixels da referência; o inpainting deixa artefatos e não recupera partes ocultas. A ordem inicial foi objeto e depois texto, como convenção, sem inferência de profundidade. O SVG mantém textos editáveis, mas não transforma a imagem inteira em vetores ou camadas originais.

O erro médio RGB foi 5,3411/255 e 14,4977/255 respectivamente. Esses números dependem de alinhamento, renderizador e de extensas áreas preservadas da referência; não são métricas de fidelidade perceptiva ou qualidade de separação. Não foram coletadas máscaras humanas para medir precisão ou recall; estes dois exemplos não representam uma avaliação geral.
