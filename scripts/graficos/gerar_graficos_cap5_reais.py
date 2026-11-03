import ast
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, 'src'))
from lang_detector import calcular_score_xi, classificar_lingua

PASTA_SAIDA = os.path.join(RAIZ, 'results', 'graficos_cap5_reais')
os.makedirs(PASTA_SAIDA, exist_ok=True)
DPI = 200

COR_PT, COR_MISTO, COR_XI = '#2a78d6', '#eb6834', '#1baf7a'
COR_ANTES, COR_DEPOIS = '#2a78d6', '#eb6834'
COR_NEG, COR_NEU, COR_POS = '#e34948', '#b9b8b2', '#2a78d6'
TINTA, TINTA_2, GRELHA = '#0b0b0b', '#52514e', '#e4e3df'

plt.rcParams.update({
    'font.size': 10.5, 'axes.edgecolor': TINTA_2, 'axes.labelcolor': TINTA,
    'xtick.color': TINTA_2, 'ytick.color': TINTA_2, 'axes.titleweight': 'bold',
    'axes.titlesize': 12.5, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
})

CONJUNTOS = ['AfriSenti\nPortuguês MZ', 'AfriSenti\nXitsonga', 'AfriEmo\nPortuguês MZ']


def limpar_texto(texto: str) -> str:
    if not isinstance(texto, str):
        return ''
    texto = re.sub(r'http\S+|www\.\S+', '', texto)
    texto = re.sub(r'@\w+', '', texto)
    texto = re.sub(r'<[^>]+>', '', texto)
    texto = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', texto)
    return re.sub(r'\s+', ' ', texto).strip()


def num_palavras(texto: str) -> int:
    return len(texto.split()) if isinstance(texto, str) else 0


def carregar_dados() -> dict:
    def afrisenti(lingua):
        partes = [pd.read_csv(os.path.join(RAIZ, f'data/raw/afrisenti_oficial/afrisenti_{lingua}_{s}.tsv'), sep='\t', dtype=str)
                  for s in ['train', 'dev', 'test']]
        return pd.concat(partes, ignore_index=True).rename(columns={'tweet': 'texto'})

    pt, ts = afrisenti('por'), afrisenti('tso')
    partes = []
    for split in ['train', 'dev', 'test']:
        df = pd.read_csv(os.path.join(RAIZ, f'brighter_data/ptmz_{split}.csv'), encoding='utf-8')
        n_bruto = len(df)
        df = df.drop_duplicates(subset='text').assign(split=split)
        print(f'[AfriEmo] {split}: {n_bruto} linhas -> {len(df)} textos únicos')
        partes.append(df)
    emo = pd.concat(partes, ignore_index=True).rename(columns={'text': 'texto'})
    dados = dict(zip(CONJUNTOS, [pt, ts, emo]))
    for nome, df in dados.items():
        df['texto_limpo'] = df['texto'].apply(limpar_texto)
        df['n_antes'] = df['texto'].apply(num_palavras)
        df['n_depois'] = df['texto_limpo'].apply(num_palavras)
    return dados


def aplicar_detector(dados: dict, lexico: set):
    for nome, df in dados.items():
        df['score_xi'] = df['texto_limpo'].apply(lambda t: calcular_score_xi(t, lexico))
        df['cat_lingua'] = df['texto_limpo'].apply(lambda t: classificar_lingua(t, lexico) if t else 'misto')
        print(f'[Detector] {nome.replace(chr(10), " ")}: {df["cat_lingua"].value_counts().to_dict()}')


def num(k: int) -> str:
    return f'{k:,}'.replace(',', '.')


def pct(p: float) -> str:
    return f'{p:.1f}'.replace('.', ',') + '%'


def grafico_5_1_categoria_linguistica(dados: dict):
    cats = [('pt', 'Português', COR_PT), ('misto', 'Código-Misto', COR_MISTO), ('xi', 'Xichangana', COR_XI)]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    y = np.arange(len(CONJUNTOS))[::-1]
    for yi, nome in zip(y, CONJUNTOS):
        df = dados[nome]
        n = len(df)
        esquerda = 0
        for cod, rot, cor in cats:
            k = int((df['cat_lingua'] == cod).sum())
            p = 100 * k / n
            ax.barh(yi, p, left=esquerda, height=0.56, color=cor, edgecolor='white', linewidth=2)
            if p >= 7:
                ax.text(esquerda + p / 2, yi, f'{pct(p)}\n(n={num(k)})', ha='center', va='center',
                        fontsize=8.5, color='white', fontweight='bold')
            esquerda += p
        ax.text(101, yi, f'N = {num(n)}', va='center', fontsize=9, color=TINTA_2)
    ax.set_yticks(y)
    ax.set_yticklabels(CONJUNTOS)
    ax.set_xlim(0, 112)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel('Percentagem de textos (%)')
    ax.xaxis.grid(True, color=GRELHA)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in cats]
    ax.legend(handles, [r for _, r, _ in cats], ncol=3, loc='lower center', bbox_to_anchor=(0.45, 1.0), frameon=False)
    ax.set_title('Categoria linguística atribuída pelo detector em dois estágios', pad=30)
    plt.tight_layout()
    caminho = os.path.join(PASTA_SAIDA, 'grafico_5_1_categoria_linguistica.png')
    plt.savefig(caminho, dpi=DPI, bbox_inches='tight')
    plt.close()
    return caminho


def grafico_5_2_comprimento(dados: dict):
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    pos = np.arange(len(CONJUNTOS)) * 1.0
    d = 0.19
    for i, nome in enumerate(CONJUNTOS):
        df = dados[nome]
        for desloc, col, cor in [(-d, 'n_antes', COR_ANTES), (d, 'n_depois', COR_DEPOIS)]:
            bp = ax.boxplot(df[col], positions=[pos[i] + desloc], widths=0.32, patch_artist=True, showfliers=False,
                            showmeans=True, medianprops={'color': 'white', 'linewidth': 2},
                            meanprops={'marker': 'D', 'markerfacecolor': 'white', 'markeredgecolor': TINTA, 'markersize': 6},
                            whiskerprops={'color': TINTA_2}, capprops={'color': TINTA_2}, boxprops={'linewidth': 0})
            bp['boxes'][0].set_facecolor(cor)
            q3 = df[col].quantile(0.75)
            iqr = q3 - df[col].quantile(0.25)
            topo = min(df[col].max(), q3 + 1.5 * iqr)
            ax.text(pos[i] + desloc, topo + 0.8, f'x̄={df[col].mean():.1f}'.replace('.', ','), ha='center', fontsize=8.5, color=TINTA)
    ax.set_xticks(pos)
    ax.set_xticklabels(CONJUNTOS)
    ax.set_ylabel('Número de palavras por texto')
    ax.yaxis.grid(True, color=GRELHA)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COR_ANTES), plt.Rectangle((0, 0), 1, 1, color=COR_DEPOIS),
               plt.Line2D([], [], marker='D', ls='', markerfacecolor='white', markeredgecolor=TINTA)]
    ax.legend(handles, ['Antes da limpeza', 'Depois da limpeza', 'Média'], ncol=3, loc='lower center',
              bbox_to_anchor=(0.5, 1.0), frameon=False)
    ax.set_title('Comprimento dos textos antes e depois do pré-processamento', pad=30)
    plt.tight_layout()
    caminho = os.path.join(PASTA_SAIDA, 'grafico_5_2_comprimento_textos.png')
    plt.savefig(caminho, dpi=DPI, bbox_inches='tight')
    plt.close()
    return caminho


def grafico_5_3_sentimento(dados: dict):
    classes = [('negative', 'Negativo', COR_NEG), ('neutral', 'Neutro', COR_NEU), ('positive', 'Positivo', COR_POS)]
    nomes = CONJUNTOS[:2]
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    y = np.arange(len(nomes))[::-1]
    for yi, nome in zip(y, nomes):
        df = dados[nome]
        n = len(df)
        esquerda = 0
        for cod, rot, cor in classes:
            k = int((df['label'] == cod).sum())
            p = 100 * k / n
            ax.barh(yi, p, left=esquerda, height=0.56, color=cor, edgecolor='white', linewidth=2)
            ax.text(esquerda + p / 2, yi, f'{pct(p)}\n(n={num(k)})', ha='center', va='center', fontsize=8.5,
                    color=TINTA if cod == 'neutral' else 'white', fontweight='bold')
            esquerda += p
        ax.text(101, yi, f'N = {num(n)}', va='center', fontsize=9, color=TINTA_2)
    ax.set_yticks(y)
    ax.set_yticklabels(nomes)
    ax.set_xlim(0, 112)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel('Percentagem de tweets (%)')
    ax.spines[['top', 'right']].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in classes]
    ax.legend(handles, [r for _, r, _ in classes], ncol=3, loc='lower center', bbox_to_anchor=(0.45, 1.0), frameon=False)
    ax.set_title('Distribuição da polaridade emocional no AfriSenti', pad=30)
    plt.tight_layout()
    caminho = os.path.join(PASTA_SAIDA, 'grafico_5_3_sentimento_afrisenti.png')
    plt.savefig(caminho, dpi=DPI, bbox_inches='tight')
    plt.close()
    return caminho


def contar_emocoes(df: pd.DataFrame) -> pd.Series:
    emocoes = {'joy': 'Alegria', 'sadness': 'Tristeza', 'anger': 'Raiva', 'fear': 'Medo', 'surprise': 'Surpresa', 'disgust': 'Nojo'}
    cont = {pt: int(df[en].sum()) for en, pt in emocoes.items()}
    cont['Neutro'] = int((df['emotions'].apply(lambda s: len(ast.literal_eval(s))) == 0).sum())
    return pd.Series(cont).sort_values()


def grafico_5_4_emocoes(dados: dict):
    df = dados[CONJUNTOS[2]]
    cont = contar_emocoes(df)
    n = len(df)
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    cores = [COR_NEU if k.startswith('Neutro') else COR_PT for k in cont.index]
    ax.barh(cont.index, cont.values, color=cores, height=0.6)
    for i, v in enumerate(cont.values):
        ax.text(v + n * 0.006, i, f'{num(v)}  ({pct(100 * v / n)})', va='center', fontsize=9, color=TINTA)
    ax.set_xlim(0, cont.max() * 1.22)
    ax.set_xlabel('Número de textos')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: num(int(v))))
    ax.xaxis.grid(True, color=GRELHA)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_title('Distribuição das emoções no AfriEmo — Português MZ', pad=12)
    plt.tight_layout()
    caminho = os.path.join(PASTA_SAIDA, 'grafico_5_4_emocoes_afriemo.png')
    plt.savefig(caminho, dpi=DPI, bbox_inches='tight')
    plt.close()
    return caminho


def escrever_resumo(dados: dict):
    linhas = ['RESUMO NUMÉRICO — CAPÍTULO 5 (dados reais)', '=' * 60]
    for nome, df in dados.items():
        rot = nome.replace('\n', ' ')
        linhas += [f'\n## {rot}  (N = {len(df)})',
                   f'  Textos com @menção: {df["texto"].str.contains("@").sum()} | com URL: {df["texto"].str.contains("http").sum()}',
                   f'  Palavras antes : média {df.n_antes.mean():.2f} | DP {df.n_antes.std():.2f} | mediana {df.n_antes.median():.0f}',
                   f'  Palavras depois: média {df.n_depois.mean():.2f} | DP {df.n_depois.std():.2f} | mediana {df.n_depois.median():.0f}',
                   f'  Redução média: {df.n_antes.mean() - df.n_depois.mean():.2f} palavras '
                   f'({100 * (1 - df.n_depois.mean() / df.n_antes.mean()):.1f}%)',
                   f'  Categoria linguística: {df.cat_lingua.value_counts().to_dict()}',
                   f'  Cobertura do léxico Xichangana (score_xi): média {df.score_xi.mean():.3f} | '
                   f'mediana {df.score_xi.median():.3f} | >=0,8: {(df.score_xi >= 0.8).sum()} | '
                   f'0,3-0,8: {((df.score_xi >= 0.3) & (df.score_xi < 0.8)).sum()} | '
                   f'0,05-0,3: {((df.score_xi >= 0.05) & (df.score_xi < 0.3)).sum()} | <0,05: {(df.score_xi < 0.05).sum()}']
        if 'label' in df:
            linhas.append(f'  Sentimento: {df.label.value_counts().to_dict()}')
        if 'emotions' in df:
            linhas.append(f'  Emoções: {contar_emocoes(df).to_dict()}')
            linhas.append(f'  Textos por split (únicos): {df.split.value_counts().to_dict()}')
    caminho = os.path.join(PASTA_SAIDA, 'resumo_numeros_cap5.txt')
    with open(caminho, 'w', encoding='utf-8') as f:
        f.write('\n'.join(linhas) + '\n')
    return caminho


if __name__ == '__main__':
    print('=' * 60)
    print('  EmoText MZ — Gráficos do Capítulo 5 (DADOS REAIS)')
    print('=' * 60)
    with open(os.path.join(RAIZ, 'data/lexicon/lexico_xichangana_set.txt'), encoding='utf-8') as f:
        LEXICO = {l.strip().lower() for l in f if l.strip()}
    print(f'[Léxico] {len(LEXICO)} termos Xichangana')
    dados = carregar_dados()
    aplicar_detector(dados, LEXICO)
    saidas = [grafico_5_1_categoria_linguistica(dados), grafico_5_2_comprimento(dados),
              grafico_5_3_sentimento(dados), grafico_5_4_emocoes(dados), escrever_resumo(dados)]
    print('\nFicheiros gerados:')
    for s in saidas:
        print('  ', s)
