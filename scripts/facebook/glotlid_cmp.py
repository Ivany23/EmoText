import csv, glob, json, re, sys, collections
import fasttext
from openpyxl import load_workbook
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, precision_recall_fscore_support
sys.path.insert(0, r'C:/Users/25884/Downloads/PFC_Last/EmoTextMZ/src')
from lang_detector import calcular_score_xi
D = r'C:/Users/25884/Downloads/PFC_Last/EmoTextMZ/data/facebook_real/'
m = fasttext.load_model(glob.glob('hf/**/model.bin', recursive=True)[0])
TSONGA = {'tso_Latn', 'tsc_Latn', 'rng_Latn'}
def glot(t, k=5):
    t = re.sub(r'\s+', ' ', t).strip()
    return [(l.replace('__label__', ''), p) for p, l in m.f.predict(t, k, 0.0, 'strict')]
lex = set(l.strip().lower() for l in open(r'C:/Users/25884/Downloads/PFC_Last/EmoTextMZ/data/lexicon/lexico_xichangana_set.txt', encoding='utf-8') if l.strip())
def palavras(t): return re.findall(r"[^\W\d_]+(?:'[^\W\d_]+)?", t)
def det_B(t):
    s = calcular_score_xi(t, lex); top = glot(t, 1)[0][0]
    if s >= 0.8: return 'xi'
    if s >= 0.3: return 'misto'
    if s < 0.05 and top == 'por_Latn': return 'pt'
    return 'misto'
def det_C(t):
    top = glot(t, 1)[0][0]
    if top in TSONGA: return 'xi'
    ws = [w for w in palavras(t) if len(w) >= 3]
    tso_w = 0
    for w in ws:
        l, p = glot(w.lower(), 1)[0]
        if l in TSONGA and p >= 0.5: tso_w += 1
    if top == 'por_Latn':
        return 'misto' if tso_w >= 1 else 'pt'
    return 'misto' if tso_w >= 1 else 'outro'
M = {'Português': 'pt', 'Mistura': 'misto', 'Xichangana': 'xi', 'Outro': 'outro'}
ws_ = load_workbook(D + 'verificacao_lingua_200_IVANY.xlsx')['Verificacao']
iv = {int(r[0]): M[r[6].strip()] for r in ws_.iter_rows(min_row=2, values_only=True) if r[0]}
am = {int(r['n']): r for r in csv.DictReader(open(D + 'amostra_200_com_detector.csv', encoding='utf-8-sig'))}
L = ['pt', 'misto', 'xi']
res = {}
for nome, f in [('A: léxico + langdetect (actual)', lambda n: am[n]['detector']),
                ('B: léxico + GlotLID', lambda n: det_B(am[n]['texto'])),
                ('C: GlotLID (texto e palavras)', lambda n: det_C(am[n]['texto']))]:
    pred = {n: f(n) for n in iv}
    ns = [n for n in iv if iv[n] != 'outro']
    y = [iv[n] for n in ns]; p = [pred[n] for n in ns]
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, labels=L, zero_division=0)
    res[nome] = {'exactidao': accuracy_score(y, p), 'macro_f1': f1_score(y, p, labels=L, average='macro', zero_division=0),
                 'kappa': cohen_kappa_score(y, p), 'prec': dict(zip(L, pr)), 'sens': dict(zip(L, rc)), 'f1': dict(zip(L, f1)),
                 'previstos': dict(collections.Counter(p)), 'outro_humano_previsto': dict(collections.Counter(pred[n] for n in iv if iv[n] == 'outro'))}
    if nome.startswith('C'):
        erros = [(iv[n], pred[n], am[n]['texto'][:90]) for n in ns if iv[n] != pred[n]]
json.dump(res, open('glotlid_resultados.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
for k, v in res.items():
    print(f"{k:34s} exact {100*v['exactidao']:.1f} | macroF1 {100*v['macro_f1']:.1f} | kappa {v['kappa']:.2f} | F1 pt {100*v['f1']['pt']:.1f} misto {100*v['f1']['misto']:.1f} xi {100*v['f1']['xi']:.1f} | prec misto {100*v['prec']['misto']:.1f} sens misto {100*v['sens']['misto']:.1f} | prev {v['previstos']}")
print('erros C (amostra):'); [print('  ', e) for e in erros[:25]]
