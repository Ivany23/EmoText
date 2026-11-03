import csv, re, json, collections, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from openpyxl import load_workbook
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, precision_recall_fscore_support, confusion_matrix
D = r'C:/Users/25884/Downloads/PFC_Last/EmoTextMZ/data/facebook_real/'
FIG = r'C:/Users/25884/Downloads/PFC_Last/Arquivo_Antigo/ArquivoFidedigno/EmoText/figuras_cap5/'
fn = D+'facebook_comentarios_anonimizados.csv'
rows = list(csv.DictReader(open(fn, encoding='utf-8-sig')))
tel = re.compile(r'(?<!\w)(?:\+?258[\s-]?)?8[2-7](?:[\s-]?\d){7}(?!\w)|(?<!\w)\d{4,}(?:[\s-]\d{3,})+(?!\w)')
nmask = 0
for r in rows:
    novo = tel.sub('[TELEFONE]', r['texto'])
    if novo != r['texto']: nmask += 1; r['texto'] = novo
with open(fn, 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
R = {'telefones_mascarados': nmask, 'n_validos': len(rows)}
R['por_pub'] = dict(collections.Counter(r['publicacao'] for r in rows))
R['det_corpus'] = dict(collections.Counter(r['detector'] for r in rows))
R['det_por_pub'] = {p: dict(collections.Counter(r['detector'] for r in rows if r['publicacao']==p)) for p in ['P1','P2','P3','P4']}
R['palavras_media'] = round(np.mean([int(r['n_palavras']) for r in rows]), 2)
R['respostas'] = sum(r['e_resposta']=='sim' for r in rows)
M = {'Português':'pt','Mistura':'misto','Xichangana':'xi','Outro':'outro'}
ws = load_workbook(D+'verificacao_lingua_200_IVANY.xlsx')['Verificacao']
iv = {}; sug = {}
for r in ws.iter_rows(min_row=2, values_only=True):
    if r[0]: iv[int(r[0])] = M[r[6].strip()]; sug[int(r[0])] = M[r[2]]
det = {int(r['n']): r['detector'] for r in csv.DictReader(open(D+'amostra_200_com_detector.csv', encoding='utf-8-sig'))}
R['alteradas'] = sum(iv[n]!=sug[n] for n in iv)
R['humano_amostra'] = dict(collections.Counter(iv.values()))
R['det_amostra'] = dict(collections.Counter(det.values()))
ns = [n for n in iv if iv[n]!='outro']; L=['pt','misto','xi']
y=[iv[n] for n in ns]; p=[det[n] for n in ns]
pr, rc, f1, sp = precision_recall_fscore_support(y, p, labels=L, zero_division=0)
R['metricas'] = {'n': len(ns), 'exactidao': accuracy_score(y,p), 'macro_f1': f1_score(y,p,labels=L,average='macro',zero_division=0), 'kappa': cohen_kappa_score(y,p),
  'por_classe': {c: {'precisao': pr[i], 'sensibilidade': rc[i], 'f1': f1[i], 'suporte': int(sp[i])} for i,c in enumerate(L)}}
R['matriz'] = confusion_matrix([iv[n] for n in iv], [det[n] for n in iv], labels=['pt','misto','xi','outro']).tolist()
Ncorp = R['det_corpus']; am = collections.Counter(det.values())
pop = collections.Counter()
for n in iv: pop[iv[n]] += Ncorp[det[n]]/am[det[n]]
tot = sum(pop.values()); R['humano_estimado_pct'] = {k: 100*v/tot for k,v in pop.items()}
R['exactidao_ponderada'] = sum(Ncorp[det[n]]/am[det[n]] for n in iv if iv[n]==det[n])/tot
rng = np.random.default_rng(42); est=[]
strata = {c:[n for n in iv if det[n]==c] for c in am}
for _ in range(2000):
    c2 = collections.Counter()
    for c, lst in strata.items():
        smp = rng.choice(lst, size=len(lst), replace=True)
        for n in smp: c2[iv[n]] += Ncorp[c]/len(lst)
    est.append(100*c2['misto']/sum(c2.values()))
R['ic95_mistura'] = [float(np.percentile(est,2.5)), float(np.percentile(est,97.5))]
json.dump(R, open('fb_resultados.json','w',encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
print(json.dumps(R, ensure_ascii=False, indent=1, default=float))
COR_PT, COR_MISTO, COR_XI, COR_OUT = '#2a78d6', '#eb6834', '#1baf7a', '#b9b8b2'
TINTA, TINTA_2, GRELHA = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 10.5, 'axes.edgecolor': TINTA_2, 'axes.labelcolor': TINTA, 'xtick.color': TINTA_2, 'ytick.color': TINTA_2,
  'axes.titleweight': 'bold', 'axes.titlesize': 12.5, 'figure.facecolor': 'white', 'axes.facecolor': 'white'})
pct = lambda v: (f'{v:.2f}' if 0 < v < 0.1 else f'{v:.1f}').replace('.', ',') + '%'
cats = [('pt','Português',COR_PT),('misto','Código-Misto',COR_MISTO),('xi','Xichangana',COR_XI),('outro','Outro',COR_OUT)]
tot_det = sum(Ncorp.values())
linhas = [('Detector\nem dois estágios', {k: 100*Ncorp.get(k,0)/tot_det for k,_,_ in cats}),
          ('Verificação humana\n(estimativa)', {k: R['humano_estimado_pct'].get(k,0) for k,_,_ in cats})]
fig, ax = plt.subplots(figsize=(9.5, 3.9)); y = [1, 0]
for yi, (nome, dist) in zip(y, linhas):
    esq = 0
    for cod, rot, cor in cats:
        v = dist[cod]
        ax.barh(yi, v, left=esq, height=0.56, color=cor, edgecolor='white', linewidth=2)
        if v >= 6: ax.text(esq+v/2, yi, pct(v), ha='center', va='center', fontsize=9, color=TINTA if cod=='outro' else 'white', fontweight='bold')
        esq += v
    peq = [f'{rot} {pct(dist[cod])}' for cod, rot, _ in cats if 0 < dist[cod] < 6]
    if peq: ax.text(100, yi+0.33, ' · '.join(peq), ha='right', va='bottom', fontsize=8.5, color=TINTA_2)
    ax.text(101, yi, 'N = 3.115' if yi==1 else 'n = 200', va='center', fontsize=9, color=TINTA_2)
ax.set_yticks(y); ax.set_yticklabels([l[0] for l in linhas]); ax.set_xlim(0,112); ax.set_xticks(range(0,101,20))
ax.set_xlabel('Percentagem de comentários (%)'); ax.xaxis.grid(True, color=GRELHA); ax.set_axisbelow(True)
ax.spines[['top','right']].set_visible(False)
ax.legend([plt.Rectangle((0,0),1,1,color=c) for _,_,c in cats], [r for _,r,_ in cats], ncol=4, loc='lower center', bbox_to_anchor=(0.45,1.0), frameon=False)
ax.set_title('Composição linguística dos comentários do Facebook', pad=30)
plt.tight_layout(); plt.savefig(FIG+'grafico_5_4_facebook_detector_humano.png', dpi=200, bbox_inches='tight'); plt.close()
