# Experimento: anúncio Inca Kola

Fonte: [Inca Kola Advertisement, Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Inca_Kola_Advertisement.png), autor desconhecido, década de 1960. A página identifica o anúncio como domínio público e explica sua base de licenciamento. A marca aparece por ser parte da referência histórica. `reference.png` é o arquivo baixado, 415 × 568 px, SHA-256 `8beb00c8c094882fbaf5f3679c8d4280f7f70728aca8c68de957b31bab308f5d`.

## Leitura visual

Fundo creme liso; selo azul no alto à esquerda; cabeçalho pequeno à direita; figura recortada centralizada ocupando aproximadamente `[61, 85, 281, 396]`; frase rosa manuscrita na base. A figura fica visualmente em frente ao fundo; a relação precisa entre partes da figura não é recuperável. A posição e estilo dos textos foram estimados, não extraídos como objetos editáveis.

O OCR detectou seis trechos. Acertou “Tome”, “INCA”, “KOLA” e “MARCA REG”; leu o cabeçalho como “mm N O T A S” e a frase como “iDe xabor Nacional”. A leitura visual é “mini NOTAS” e “¡De sabor nacional!”. Isso mostra que a confiança de 0,9443 na frase não equivale a correção. Veja `ocr/ocr.json` e `ocr/ocr-overlay.png`.

A segmentação em oito cores encontrou, entre outras, uma região de fundo e várias regiões fragmentadas da pessoa. A máscara GrabCut foi inicializada com `[57, 76, 290, 410]`; cobre a pessoa e o tecido, com limites ainda sujeitos a erro. O sujeito foi reaproveitado do próprio original: não é uma reconstrução independente da fotografia e não prova recuperação de pixels ocultos. Veja `segments/subject.png`, `segments/subject-mask.png` e `segments/segments-overlay.png`.

## Reconstrução e avaliação

`reconstruct.py` redesenha fundo, círculos e textos como primitivas, compõe o sujeito recortado e salva `reconstruction.png`. As fontes são aproximações e variam conforme o sistema. A frase inferior, por exemplo, tem desenho e posição diferentes da caligrafia da referência; o selo azul também perde textura e proporções de letras. O fundo amplo e o reaproveitamento do sujeito tornam o erro médio RGB de **6,6183/255** aparentemente baixo: esse valor é sensível a alinhamento e não mede qualidade perceptiva. Veja `comparison/overlay.png` e `comparison/difference.png`. A recriação é raster gerada por código, não um PSD/SVG com camadas editáveis; o código mantém os elementos geométricos e textos alteráveis.

Reproduzir a partir da raiz do repositório, após instalar `.[vision]`:

```bash
rea-design inspect examples/vintage-ad/reference.png --out examples/vintage-ad/evidence
rea-design ocr examples/vintage-ad/reference.png --out examples/vintage-ad/ocr
rea-design segment examples/vintage-ad/reference.png --out examples/vintage-ad/segments --rect 57 76 290 410 --min-area 0.004
python examples/vintage-ad/reconstruct.py
rea-design compare examples/vintage-ad/reference.png examples/vintage-ad/reconstruction.png --out examples/vintage-ad/comparison
```

Próximos testes necessários: outros idiomas, tipografia inclinada, elementos sobrepostos e imagens sem fundo uniforme; comparação de máscaras com anotações humanas; revisão das fontes e máscara do sujeito. Este caso não mede desempenho geral.
