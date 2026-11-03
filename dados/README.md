# Dados

Os dados não estão incluídos neste repositório.

## AfriSenti (polaridade)

Ficheiros oficiais da SemEval-2023 Task 12, subconjuntos `por` (português de Moçambique) e `tso` (xitsonga):

https://github.com/afrisenti-semeval/afrisent-semeval-2023/tree/main/data

Colocar em `dados/afrisenti/` os ficheiros `train.tsv`, `dev.tsv` e `test.tsv` de cada língua.

## AfriEmo / BRIGHTER (emoções)

Conjunto `brighter-dataset/BRIGHTER-emotion-categories`, configuração `ptmz`, no Hugging Face. Pode ser descarregado com:

```
python scripts/download_brighter.py
```

Os ficheiros de validação e de teste contêm cada texto duas vezes; os cadernos removem os duplicados.

## Léxico NRC

Tradução portuguesa do NRC Emotion Lexicon (v0.92), disponível em https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm. O caderno descarrega-o automaticamente.

## Comentários do Facebook

5.000 comentários exportados de quatro publicações públicas (Outubro de 2026). Não são publicados, por conterem nomes e outros dados pessoais. Os scripts em `scripts/facebook/` documentam a pseudonimização, a limpeza, a amostragem e a avaliação do detector.
