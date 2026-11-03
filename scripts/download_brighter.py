import os
import pandas as pd
from datasets import load_dataset, get_dataset_config_names

print("Buscando dados no dataset BRIGHTER (SemEval 2025)...")

languages_to_download = ['ptmz']
xitsonga_candidates = ['tso', 'ts', 'xitsonga']

output_dir = "brighter_data"
os.makedirs(output_dir, exist_ok=True)

try:
    configs = get_dataset_config_names('brighter-dataset/BRIGHTER-emotion-categories')
    print("Idiomas disponíveis:", configs)

    langs = []
    if 'ptmz' in configs:
        langs.append('ptmz')

    for cand in xitsonga_candidates:
        if cand in configs:
            langs.append(cand)
            print(f"Xitsonga encontrado sob o código: {cand}")
            break

    for lang in langs:
        print(f"\nBaixando dados para a língua: {lang}...")
        dataset = load_dataset('brighter-dataset/BRIGHTER-emotion-categories', lang)

        for split in dataset.keys():
            df = pd.DataFrame(dataset[split])
            filename = os.path.join(output_dir, f"{lang}_{split}.csv")
            df.to_csv(filename, index=False, encoding='utf-8')
            print(f"-> Salvo {len(df)} exemplos em {filename}")

except Exception as e:
    print(f"Erro ao baixar dataset: {e}")
