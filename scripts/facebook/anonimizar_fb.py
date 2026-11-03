import csv, hashlib, re, sys, random, collections
sys.path.insert(0, r'C:/Users/25884/Downloads/PFC_Last/EmoTextMZ/src')
from lang_detector import classificar_lingua, calcular_score_xi
csv.field_size_limit(10**8)
FONTES = {
 '1016312621295973': r'C:/Users/25884/Downloads/Comments Exporter-1016312621295973-1809-comments-replies-allComments.csv',
 '3074558866069881': r'C:/Users/25884/Downloads/Comments Exporter-3074558866069881-471-comments-replies-allComments.csv',
 '1124126210543290': r'C:/Users/25884/Downloads/Comments Exporter-1124126210543290-1179-comments-replies-allComments.csv',
 '1629487814815526': r'C:/Users/25884/OneDrive/Desktop/Comments Exporter-1629487814815526-1541-comments-replies-allComments.csv',
}
OUT = r'C:/Users/25884/Downloads/PFC_Last/EmoTextMZ/data/facebook_real/'
h = lambda s: hashlib.sha256(s.encode()).hexdigest()[:16]
def limpar(t):
    t = re.sub(r'https?://\S+|www\.\S+', ' ', t)
    t = re.sub(r'@\S+', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()
lex = set(l.strip().lower() for l in open(r'C:/Users/25884/Downloads/PFC_Last/EmoTextMZ/data/lexicon/lexico_xichangana_set.txt', encoding='utf-8') if l.strip())
linhas, vistos = [], set()
stats = collections.Counter()
for k, (pid, f) in enumerate(FONTES.items(), 1):
    for r in csv.DictReader(open(f, encoding='utf-8-sig', newline='')):
        stats['bruto'] += 1
        txt = limpar(r['Comment'] or '')
        if not txt: stats['vazio'] += 1; continue
        chave = txt.lower()
        if chave in vistos: stats['duplicado'] += 1; continue
        vistos.add(chave)
        n_pal = len(re.findall(r"[^\W\d_]{2,}", txt))
        if n_pal < 3: stats['curto'] += 1; continue
        linhas.append({'comment_id': h(r['Comment ID']), 'publicacao': f'P{k}', 'post_id': h(pid),
                       'data': r['Comment Time'], 'gostos': r['Like Count'], 'e_resposta': 'sim' if r['Reply To Comment'].strip() else 'nao',
                       'texto': txt, 'n_palavras': n_pal})
for x in linhas:
    x['detector'] = classificar_lingua(x['texto'], lex)
    x['score_xi'] = round(calcular_score_xi(x['texto'], lex), 3)
cols = list(linhas[0].keys())
with open(OUT+'facebook_comentarios_anonimizados.csv', 'w', encoding='utf-8-sig', newline='') as fo:
    w = csv.DictWriter(fo, fieldnames=cols); w.writeheader(); w.writerows(linhas)
print(dict(stats), 'finais', len(linhas))
print('por publicacao', collections.Counter(x['publicacao'] for x in linhas))
print('detector', collections.Counter(x['detector'] for x in linhas))
