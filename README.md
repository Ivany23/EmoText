# EmoText MZ

Processamento de Linguagem Natural para Análise Emocional em Textos Digitais Moçambicanos em Português e Xichangana.

Projecto Final do Curso de Licenciatura em Engenharia e Ciências Computacionais — Instituto Superior de Transportes e Comunicações (ISUTC).

- **Autor:** Ivany Mazia Massinga (LECC51/5976)
- **Supervisor:** Eng. Ivan Ruby
- **Ano:** 2026

## Objectivo

Avaliar e adaptar modelos de PLN para a análise emocional de textos digitais moçambicanos em dois níveis:

1. **Polaridade** (positivo, negativo, neutro) em português de Moçambique e em xitsonga (variedades changana e ronga), com o AfriSenti.
2. **Emoções discretas** (alegria, tristeza, raiva, medo, surpresa, nojo) em português de Moçambique, com o AfriEmo (BRIGHTER, SemEval-2025 Task 11).

Inclui ainda a caracterização de 3.115 comentários públicos do Facebook e a verificação manual do detector de língua numa amostra de 200 comentários.

## Estrutura

```
notebooks/        Cadernos do Google Colab usados no treino e na avaliação
src/              Módulos do protótipo: limpeza, detector de língua e avaliação
lexico/           Léxico de xichangana usado pelo detector de língua
scripts/
  download_brighter.py   Descarga do AfriEmo (BRIGHTER)
  graficos/              Gráficos e figuras do Capítulo 5
  facebook/              Pseudonimização, amostragem e avaliação do detector nos comentários do Facebook
  estatistica/           Intervalos de confiança por bootstrap
resultados/
  execucao_polaridade/   Tabelas e registos da polaridade, do zero-shot e da configuração inicial das emoções
  execucao_emocoes/      Tabelas e registos da detecção de emoções
  facebook/              Avaliação do detector face à verificação manual e comparação com o GlotLID
  estatistica/           Intervalos de confiança e testes de bootstrap emparelhado
  figuras/               Gráficos e figuras do Capítulo 5
dados/            Instruções para obter os conjuntos de dados
```

## Execução

Os cadernos foram executados no Google Colab com uma GPU NVIDIA Tesla T4:

- Caderno de detecção de emoções: https://colab.research.google.com/drive/1P_c1KMxGZrgkKRZQ4JvMPAJEffWumnfq

Passos:

1. Obter os dados conforme `dados/README.md`.
2. Abrir o caderno no Colab, escolher **Ambiente de execução → GPU T4** e executar todas as células.
3. Os resultados são gravados nas pastas `tabelas/`, `figuras/` e `logs/`.

Configuração principal: comprimento máximo de 128 subpalavras, sementes 42, 43 e 44, selecção da melhor época pelo Macro-F1 na validação. Polaridade: taxa de aprendizagem 2×10⁻⁵, 3 épocas (português) e 5 épocas (xitsonga). Emoções: taxa de aprendizagem 3×10⁻⁵, 10 épocas, perda ponderada por emoção e limiares calibrados na validação.

## Resultados principais (Macro-F1 no teste, média de 3 sementes)

| Tarefa | Melhor modelo | Macro-F1 | IC 95% |
|---|---|---|---|
| Polaridade, português de Moçambique | XLM-R | 62,8% | 61,3–64,3 |
| Polaridade, xitsonga | TF-IDF + Regressão Logística | 44,8% | 39,0–50,5 |
| Transferência zero-shot português → xitsonga | XLM-R | 26,4% | — |
| Emoções, português de Moçambique | XLM-R | 40,5% | 36,7–43,8 |

Detector de língua nos comentários do Facebook, face à verificação manual (188 comentários): exactidão de 61,2% e Kappa de Cohen de 0,21.

## Dados

Os conjuntos AfriSenti e AfriEmo não são redistribuídos, de acordo com as condições de uso definidas pelos seus autores. Os comentários do Facebook também não são publicados, por conterem dados pessoais. Ver `dados/README.md`.

## Referências dos conjuntos de dados

- Muhammad, S. H. et al. 2023. AfriSenti: A Twitter Sentiment Analysis Benchmark for African Languages. EMNLP 2023.
- Muhammad, S. H. et al. 2025. BRIGHTER: BRIdging the Gap in Human-Annotated Textual Emotion Recognition Datasets for 28 Languages. ACL 2025.
