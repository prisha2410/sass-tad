from src.utils.data_loader import load_pan_labeled, MuLDDataset, split_documents, dataset_stats

print('=' * 60)
for year in ['2023', '2024', '2025']:
    print('--- PAN ' + year + ' ---')
    for tier in ['easy', 'medium', 'hard']:
        train, val, test = load_pan_labeled('data/raw', year=year, tier=tier)
        all_docs = train + val + test
        s = dataset_stats(all_docs)
        print('  ' + tier + ': ' + str(len(all_docs)) + ' docs | change_rate=' + str(s['change_rate']) + ' | avg_sents=' + str(s['avg_sentences']))

print('--- MuLD AO3 ---')
docs = MuLDDataset('data/raw/muld_ao3').load()
train, val, test = split_documents(docs)
s = dataset_stats(docs)
print('  total=' + str(len(docs)) + ' | change_rate=' + str(s['change_rate']) + ' | avg_sents=' + str(s['avg_sentences']))
