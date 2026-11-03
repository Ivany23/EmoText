import os, glob, json, unicodedata, numpy as np, pandas as pd
from sklearn.metrics import f1_score
R1 = r'C:/Users/25884/Downloads/resultados_EmoText (1)/previsoes'
R2 = r'C:/Users/25884/Downloads/resultados_emocoes_v2 (2)/previsoes'
nf = lambda s: unicodedata.normalize('NFC', s)
def ficheiros(pasta, tarefa, modelo):
    out = []
    for f in os.listdir(pasta):
        g = nf(f)
        if g.startswith(tarefa + '__' + modelo + '__s'):
            out.append(os.path.join(pasta, f))
    return sorted(out)
B, rng = 2000, np.random.default_rng(42)
EMO = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
def carregar_senti(pasta, tarefa, modelo):
    fs = ficheiros(pasta, tarefa, modelo)
    dfs = [pd.read_csv(f, encoding='utf-8-sig') for f in fs]
    y = dfs[0]['real'].to_numpy(dtype=str)
    for d in dfs: assert (d['real'].to_numpy(dtype=str) == y).all()
    return y, [d['previsto'].to_numpy(dtype=str) for d in dfs]
def carregar_emo(pasta, tarefa, modelo):
    fs = ficheiros(pasta, tarefa, modelo)
    dfs = [pd.read_csv(f, encoding='utf-8-sig') for f in fs]
    Y = dfs[0][[f'real_{e}' for e in EMO]].values
    return Y, [d[[f'prev_{e}' for e in EMO]].values for d in dfs]
LAB = np.array(['negative', 'neutral', 'positive'])
def _f1(tp, fp, fn):
    den = 2*tp + fp + fn
    return np.where(den > 0, 2*tp / np.maximum(den, 1), 0.0)
def mf1_senti(y, p, idx=None):
    if idx is not None: y, p = y[idx], p[idx]
    Y = (y[:, None] == LAB[None, :]); Pp = (p[:, None] == LAB[None, :])
    tp = (Y & Pp).sum(0); fp = (~Y & Pp).sum(0); fn = (Y & ~Pp).sum(0)
    return float(_f1(tp, fp, fn).mean())
def mf1_emo(Y, P, idx=None):
    if idx is not None: Y, P = Y[idx], P[idx]
    Y = Y.astype(bool); P = P.astype(bool)
    tp = (Y & P).sum(0); fp = (~Y & P).sum(0); fn = (Y & ~P).sum(0)
    return float(_f1(tp, fp, fn).mean())
def analisar(tipo, pasta, tarefa, modelos):
    carregar = carregar_senti if tipo == 's' else carregar_emo
    mf = mf1_senti if tipo == 's' else mf1_emo
    dados = {m: carregar(pasta, tarefa, m) for m in modelos}
    n = len(next(iter(dados.values()))[0])
    idxs = [rng.integers(0, n, n) for _ in range(B)]
    res, amostras = {}, {}
    for m, (y, ps) in dados.items():
        pont = np.mean([mf(y, p) for p in ps])
        bs = np.array([np.mean([mf(y, p, ix) for p in ps]) for ix in idxs])
        amostras[m] = bs
        res[m] = {'macro_f1': 100*pont, 'ic_inf': 100*np.percentile(bs, 2.5), 'ic_sup': 100*np.percentile(bs, 97.5), 'n_sementes': len(ps)}
    return res, amostras
def comparar(am, a, b):
    d = am[a] - am[b]
    return {'diferenca_media': 100*float(np.mean(d)), 'ic_inf': 100*float(np.percentile(d, 2.5)), 'ic_sup': 100*float(np.percentile(d, 97.5)), 'p_bilateral': float(min(1, 2*min(np.mean(d <= 0), np.mean(d >= 0))))}
out = {}
r, am = analisar('s', R1, 'senti_por', ['XLM-R', 'AfroXLMR', 'mBERT', 'TF-IDF+RL', 'Léxico NRC', 'Maioritária'])
out['polaridade_pt'] = r; out['comp_pt'] = {'XLM-R vs AfroXLMR': comparar(am, 'XLM-R', 'AfroXLMR'), 'XLM-R vs TF-IDF': comparar(am, 'XLM-R', 'TF-IDF+RL'), 'mBERT vs TF-IDF': comparar(am, 'mBERT', 'TF-IDF+RL')}
r, am = analisar('s', R1, 'senti_tso', ['TF-IDF+RL', 'AfroXLMR', 'mBERT', 'AfroXLMR-76L', 'XLM-R', 'Léxico NRC', 'Maioritária'])
out['polaridade_xi'] = r; out['comp_xi'] = {'TF-IDF vs AfroXLMR': comparar(am, 'TF-IDF+RL', 'AfroXLMR'), 'TF-IDF vs mBERT': comparar(am, 'TF-IDF+RL', 'mBERT'), 'AfroXLMR vs XLM-R': comparar(am, 'AfroXLMR', 'XLM-R'), 'mBERT vs XLM-R': comparar(am, 'mBERT', 'XLM-R')}
r, am = analisar('e', R2, 'emo_ptmz', ['XLM-R', 'AfroXLMR', 'mBERT', 'TFIDF_RL', 'Lexico_NRC'])
out['emocoes'] = r; out['comp_emo'] = {'XLM-R vs AfroXLMR': comparar(am, 'XLM-R', 'AfroXLMR'), 'XLM-R vs mBERT': comparar(am, 'XLM-R', 'mBERT'), 'XLM-R vs TF-IDF': comparar(am, 'XLM-R', 'TFIDF_RL')}
json.dump(out, open('bootstrap_resultados.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
for k, v in out.items():
    print('==', k)
    for m, x in v.items(): print(f'   {m:22s}', {kk: round(vv, 1) if isinstance(vv, float) else vv for kk, vv in x.items()})
