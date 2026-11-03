import csv, random
D = r'C:/Users/25884/Downloads/PFC_Last/EmoTextMZ/data/facebook_real/'
rows = list(csv.DictReader(open(D+'facebook_comentarios_anonimizados.csv', encoding='utf-8-sig')))
random.seed(42)
g = {c: [r for r in rows if r['detector']==c] for c in ('pt','misto','xi')}
am = g['xi'] + random.sample(g['misto'], 100) + random.sample(g['pt'], 99)
random.shuffle(am)
for i, r in enumerate(am, 1): r['n'] = i
with open(D+'amostra_200_com_detector.csv','w',encoding='utf-8-sig',newline='') as f:
    w = csv.DictWriter(f, fieldnames=['n']+list(rows[0].keys())); w.writeheader(); w.writerows(am)
for r in am: print(r['n'], '|', r['texto'][:220].replace('\n',' '))
