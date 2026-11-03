import glob, json, os, unicodedata
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import confusion_matrix

R1 = r'C:\Users\25884\Downloads\resultados_EmoText (1)'
OUT = 'figuras_cap5'
SENT = ['negative', 'neutral', 'positive']
ROT = ['Negativo', 'Neutro', 'Positivo']
plt.rcParams.update({'font.size': 10.5, 'axes.titleweight': 'bold', 'figure.facecolor': 'white'})
ex = pd.DataFrame([json.load(open(f, encoding='utf-8')) for f in glob.glob(os.path.join(R1, 'execucoes', '*.json'))])
ex['modelo'] = ex['modelo'].map(lambda s: unicodedata.normalize('NFC', s))
virg = lambda x, c=1: f'{x:.{c}f}'.replace('.', ',')

modelos = ['Léxico NRC', 'TF-IDF+RL', 'mBERT', 'XLM-R', 'AfroXLMR', 'AfroXLMR-76L']
cores = ['#b9b8b2', '#52514e', '#eb6834', '#1baf7a', '#2a78d6', '#4a3aa7']
grupos = [('senti_por', 'Português MZ'), ('senti_tso', 'Xitsonga'), ('zeroshot_por_tso', 'Zero-shot PT-MZ → Xitsonga')]
fig, ax = plt.subplots(figsize=(10, 4.8))
x = np.arange(len(grupos)); larg = 0.8 / len(modelos)
for i, (m, c) in enumerate(zip(modelos, cores)):
    med, err = [], []
    for t, _ in grupos:
        g = ex[(ex.tarefa == t) & (ex.modelo == m)]['macro_f1']
        med.append(g.mean() * 100 if len(g) else np.nan)
        err.append(g.std(ddof=1) * 100 if len(g) > 1 else np.nan)
    ax.bar(x + (i - (len(modelos) - 1) / 2) * larg, med, larg * 0.92, yerr=err, capsize=3, color=c, label=m,
           error_kw={'elinewidth': 1, 'ecolor': '#0b0b0b'})
ax.set_xticks(x); ax.set_xticklabels([g[1] for g in grupos])
ax.set_ylabel('Macro-F1 (%)'); ax.set_ylim(0, 100)
ax.yaxis.grid(True, color='#e4e3df'); ax.set_axisbelow(True); ax.spines[['top', 'right']].set_visible(False)
ax.legend(ncol=6, loc='lower center', bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=9.5)
ax.set_title('Macro-F1 na classificação da polaridade por modelo', pad=30)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'grafico_5_4_macro_f1_polaridade.png'), dpi=200, bbox_inches='tight'); plt.close()

prev = {unicodedata.normalize('NFC', os.path.basename(f)): f for f in glob.glob(os.path.join(R1, 'previsoes', '*.csv'))}
fig, eixos = plt.subplots(1, 3, figsize=(14, 4.6))
for ax, (t, m, tit) in zip(eixos, [('senti_por', 'XLM-R', 'Português MZ'), ('senti_tso', 'AfroXLMR', 'Xitsonga'),
                                     ('zeroshot_por_tso', 'XLM-R', 'Zero-shot PT-MZ → Xitsonga')]):
    fs = sorted(v for k, v in prev.items() if k.startswith(f'{t}__{m}__s'))
    media = sum(confusion_matrix(pd.read_csv(f)['real'], pd.read_csv(f)['previsto'], labels=SENT) for f in fs) / len(fs)
    pct = media / media.sum(axis=1, keepdims=True) * 100
    anot = np.array([[f'{virg(pct[i, j])}%\n({virg(media[i, j])})' for j in range(3)] for i in range(3)])
    sns.heatmap(pct, annot=anot, fmt='', cmap='Blues', vmin=0, vmax=100, cbar=False, xticklabels=ROT, yticklabels=ROT,
                linewidths=2, linecolor='white', ax=ax, annot_kws={'fontsize': 9.5})
    ax.set_xlabel('Classe prevista'); ax.set_ylabel('Classe real'); ax.set_title(f'{tit}\n{m}', fontsize=11)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'figura_5_1_matrizes_polaridade.png'), dpi=200, bbox_inches='tight'); plt.close()
print('ok')
