import io
import json
import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, 'src'))
from cleaner import limpar_texto
from lang_detector import calcular_score_xi, carregar_lexico, classificar_lingua

PASTA_MODELOS = os.environ.get('EMOTEXT_MODELOS', os.path.join(RAIZ, 'modelos'))
MODELO_POL_PT = os.path.join(PASTA_MODELOS, 'polaridade_pt_xlmr')
MODELO_EMO_PT = os.path.join(PASTA_MODELOS, 'emocoes_pt_xlmr')
MODELO_POL_XI = os.path.join(PASTA_MODELOS, 'polaridade_xi_tfidf')
LEXICO = os.path.join(RAIZ, 'lexico', 'lexico_xichangana_set.txt')

LINGUAS = {'pt': 'Português', 'misto': 'Código-Misto', 'xi': 'Xichangana'}
POLARIDADES = {'negative': 'Negativo', 'neutral': 'Neutro', 'positive': 'Positivo'}
EMOCOES = {'anger': 'Raiva', 'disgust': 'Nojo', 'fear': 'Medo', 'joy': 'Alegria', 'sadness': 'Tristeza', 'surprise': 'Surpresa'}
CORES_LINGUA = {'Português': '#2a78d6', 'Código-Misto': '#eb6834', 'Xichangana': '#1baf7a'}
CORES_POLARIDADE = {'Negativo': '#e34948', 'Neutro': '#b9b8b2', 'Positivo': '#2a78d6'}
MIN_PALAVRAS = 3

st.set_page_config(page_title='EmoText MZ', page_icon='💬', layout='wide')


@st.cache_resource(show_spinner='A carregar os modelos...')
def carregar_modelos():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    modelos = {'lexico': carregar_lexico(LEXICO), 'pol_xi': joblib.load(os.path.join(MODELO_POL_XI, 'modelo.joblib'))}
    for chave, pasta in [('pol_pt', MODELO_POL_PT), ('emo_pt', MODELO_EMO_PT)]:
        tok = AutoTokenizer.from_pretrained(pasta)
        modelo = AutoModelForSequenceClassification.from_pretrained(pasta, dtype=torch.float32).eval()
        info = json.load(open(os.path.join(pasta, 'emotext.json'), encoding='utf-8'))
        modelos[chave] = (tok, modelo, info)
    return modelos


def modelos_em_falta():
    falta = []
    for nome, pasta in [('Polaridade em português (XLM-R)', MODELO_POL_PT), ('Emoções em português (XLM-R)', MODELO_EMO_PT)]:
        if not os.path.exists(os.path.join(pasta, 'config.json')):
            falta.append(f'{nome}: {pasta}')
    if not os.path.exists(os.path.join(MODELO_POL_XI, 'modelo.joblib')):
        falta.append(f'Polaridade em xitsonga (TF-IDF): {MODELO_POL_XI}')
    return falta


def logits(tok, modelo, textos, lote=32):
    import torch
    saidas = []
    with torch.no_grad():
        for i in range(0, len(textos), lote):
            cod = tok(textos[i:i + lote], truncation=True, max_length=128, padding=True, return_tensors='pt')
            saidas.append(modelo(**cod).logits.numpy())
    return np.vstack(saidas) if saidas else np.zeros((0, modelo.config.num_labels))


def ler_csv(bruto):
    try:
        texto = bruto.decode('utf-8-sig')
    except UnicodeDecodeError:
        texto = bruto.decode('latin-1')
    cabecalho = texto.split('\n', 1)[0]
    separador = ';' if cabecalho.count(';') > cabecalho.count(',') else ','
    return pd.read_csv(io.StringIO(texto), sep=separador, dtype=str)


def preparar(df, coluna):
    total = len(df)
    dados = pd.DataFrame({'texto_original': df[coluna].fillna('').astype(str)})
    dados['texto'] = dados['texto_original'].apply(limpar_texto)
    vazios = int((dados['texto'] == '').sum())
    dados = dados[dados['texto'] != '']
    antes = len(dados)
    dados = dados.drop_duplicates(subset='texto')
    duplicados = antes - len(dados)
    antes = len(dados)
    dados = dados[dados['texto'].str.split().str.len() >= MIN_PALAVRAS].reset_index(drop=True)
    curtos = antes - len(dados)
    return dados, {'total': total, 'vazios': vazios, 'duplicados': duplicados, 'curtos': curtos, 'validos': len(dados)}


def analisar(dados, modelos, barra):
    textos = dados['texto'].tolist()
    dados['lingua'] = [classificar_lingua(t, modelos['lexico']) for t in textos]
    barra.progress(0.2, 'Língua detectada')
    dados['polaridade'] = ''
    xi = dados['lingua'] == 'xi'
    if xi.any():
        prev = modelos['pol_xi'].predict(dados.loc[xi, 'texto'].tolist())
        dados.loc[xi, 'polaridade'] = [['negative', 'neutral', 'positive'][int(p)] for p in prev]
    if (~xi).any():
        tok, modelo, info = modelos['pol_pt']
        prev = logits(tok, modelo, dados.loc[~xi, 'texto'].tolist()).argmax(axis=1)
        dados.loc[~xi, 'polaridade'] = [info['classes'][int(p)] for p in prev]
    barra.progress(0.6, 'Polaridade classificada')
    dados['emocoes'] = ''
    if (~xi).any():
        tok, modelo, info = modelos['emo_pt']
        prob = 1 / (1 + np.exp(-logits(tok, modelo, dados.loc[~xi, 'texto'].tolist())))
        limiares = np.array([info['limiares'][e] for e in info['emocoes']])
        marcadas = prob >= limiares
        dados.loc[~xi, 'emocoes'] = [', '.join(EMOCOES[e] for e, m in zip(info['emocoes'], linha) if m) or 'Nenhuma' for linha in marcadas]
    dados.loc[xi, 'emocoes'] = 'Não avaliado'
    barra.empty()
    return dados.assign(lingua=dados['lingua'].map(LINGUAS), polaridade=dados['polaridade'].map(POLARIDADES))


def grafico_barras(contagem, cores, titulo, rotulo_x, total=None):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    n = total if total is not None else contagem.sum()
    barras = ax.barh(contagem.index[::-1], contagem.values[::-1], color=[cores.get(k, '#2a78d6') for k in contagem.index[::-1]], height=0.6)
    for b, v in zip(barras, contagem.values[::-1]):
        ax.text(b.get_width() + n * 0.01, b.get_y() + b.get_height() / 2, f'{v} ({100 * v / max(n, 1):.1f}%)'.replace('.', ','), va='center', fontsize=9)
    ax.set_title(titulo, fontsize=11, fontweight='bold', loc='left')
    ax.set_xlabel(rotulo_x)
    ax.set_xlim(0, contagem.max() * 1.3 if len(contagem) else 1)
    ax.spines[['top', 'right']].set_visible(False)
    ax.xaxis.grid(True, color='#e4e3df')
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


st.title('EmoText MZ')
st.caption('Análise emocional de textos digitais moçambicanos: língua, polaridade e emoções')
st.info('Os resultados são estimativas automáticas para análise agregada. Não devem ser usados para tomar decisões sobre pessoas.')

falta = modelos_em_falta()
with st.sidebar:
    st.header('Modelos')
    st.markdown('- Língua: léxico de xichangana + langdetect\n- Polaridade (português): XLM-R ajustado\n- Polaridade (xichangana): TF-IDF + Regressão Logística\n- Emoções (português): XLM-R ajustado')
    if falta:
        st.error('Modelos em falta:\n\n' + '\n\n'.join(falta))
    else:
        st.success('Todos os modelos disponíveis')

aba1, aba2, aba3 = st.tabs(['1. Carregar ficheiro', '2. Resultados por texto', '3. Distribuição agregada'])

with aba1:
    st.subheader('Carregar ficheiro de textos')
    ficheiro = st.file_uploader('Ficheiro CSV com uma coluna de texto', type=['csv'])
    if ficheiro is not None:
        df = ler_csv(ficheiro.getvalue())
        colunas = list(df.columns)
        padrao = next((i for i, c in enumerate(colunas) if c.lower() in ('texto', 'text', 'comentario', 'comment', 'tweet')), 0)
        coluna = st.selectbox('Coluna com o texto', colunas, index=padrao)
        c1, c2 = st.columns(2)
        c1.metric('Linhas no ficheiro', f'{len(df):,}'.replace(',', '.'))
        c2.metric('Coluna seleccionada', coluna)
        st.markdown('**Pré-visualização**')
        st.dataframe(df[[coluna]].head(10), use_container_width=True, hide_index=True)
        if st.button('Analisar', type='primary', disabled=bool(falta)):
            dados, resumo = preparar(df, coluna)
            barra = st.progress(0.0, 'A analisar...')
            st.session_state['resultados'] = analisar(dados, carregar_modelos(), barra)
            st.session_state['resumo'] = resumo
            st.session_state['ficheiro'] = ficheiro.name
        if 'resumo' in st.session_state and st.session_state.get('ficheiro') == ficheiro.name:
            r = st.session_state['resumo']
            st.success(f'Análise concluída: {r["validos"]} textos analisados. Consulte os separadores 2 e 3.')
            st.caption(f'{r["total"]} linhas: {r["vazios"]} vazias, {r["duplicados"]} repetidas e {r["curtos"]} com menos de {MIN_PALAVRAS} palavras foram excluídas.')

with aba2:
    st.subheader('Resultados por texto')
    if 'resultados' not in st.session_state:
        st.info('Carregue e analise um ficheiro no separador 1.')
    else:
        res = st.session_state['resultados']
        c1, c2 = st.columns(2)
        f_lingua = c1.multiselect('Língua', list(LINGUAS.values()), default=list(LINGUAS.values()))
        f_pol = c2.multiselect('Polaridade', list(POLARIDADES.values()), default=list(POLARIDADES.values()))
        vista = res[res['lingua'].isin(f_lingua) & res['polaridade'].isin(f_pol)]
        st.caption(f'{len(vista)} de {len(res)} textos')
        st.dataframe(vista[['texto', 'lingua', 'polaridade', 'emocoes']].rename(
            columns={'texto': 'Texto', 'lingua': 'Língua', 'polaridade': 'Polaridade', 'emocoes': 'Emoções'}),
            use_container_width=True, hide_index=True, height=480)
        st.download_button('Exportar resultados (CSV)', res[['texto', 'lingua', 'polaridade', 'emocoes']].to_csv(index=False).encode('utf-8-sig'),
                           file_name='emotext_resultados.csv', mime='text/csv')

with aba3:
    st.subheader('Distribuição agregada')
    if 'resultados' not in st.session_state:
        st.info('Carregue e analise um ficheiro no separador 1.')
    else:
        res = st.session_state['resultados']
        n = len(res)
        pol = res['polaridade'].value_counts().reindex(list(POLARIDADES.values()), fill_value=0)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Textos analisados', n)
        for col, nome in zip([c2, c3, c4], ['Positivo', 'Negativo', 'Neutro']):
            col.metric(nome, f'{100 * pol[nome] / max(n, 1):.1f}%'.replace('.', ','))
        g1, g2 = st.columns(2)
        lin = res['lingua'].value_counts().reindex(list(LINGUAS.values()), fill_value=0)
        g1.pyplot(grafico_barras(lin, CORES_LINGUA, 'Língua detectada', 'Número de textos'))
        g2.pyplot(grafico_barras(pol, CORES_POLARIDADE, 'Polaridade', 'Número de textos'))
        avaliados = res[res['emocoes'] != 'Não avaliado']
        emo = pd.Series({nome: int(avaliados['emocoes'].str.contains(nome).sum()) for nome in EMOCOES.values()})
        emo['Nenhuma'] = int((avaliados['emocoes'] == 'Nenhuma').sum())
        emo = emo.sort_values(ascending=False)
        g3, g4 = st.columns(2)
        g3.pyplot(grafico_barras(emo, {}, f'Emoções ({len(avaliados)} textos avaliados)', 'Número de textos', total=len(avaliados)))
        g4.markdown('**Como ler este gráfico**')
        g4.markdown('As percentagens são calculadas sobre os textos avaliados. Como um texto pode ter mais do que uma emoção, '
                    'a soma pode ultrapassar 100%. *Nenhuma* indica textos sem nenhuma das seis emoções acima do limiar calibrado.')
        st.caption('Um texto pode ter mais do que uma emoção. As emoções não são avaliadas nos textos em xichangana, por falta de dados anotados nesta língua.')
