import pandas as pd
import numpy as np
from sklearn.manifold import TSNE

filename = '/data/bogdan/ubuntu-corpus/embeddings.txt'
data = pd.read_csv(filename, sep=' ')
filename = '/data/bogdan/ubuntu-corpus/word_counts.txt'
counts = pd.read_csv(filename, sep='\t')

counts.columns = ['word', 'count']
counts = counts.iloc[:10000,:]
data.columns = ['word'] + ['dim_%d' % i for i in range(100)] + ['dummy']
counts.set_index('word')
data.set_index('word')
data = pd.merge(data, counts)
print('Data successfully merged.')

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(data.iloc[:,1:101])

df_tsne = data.copy()
df_tsne['x-tsne'] = tsne_results[:, 0]
df_tsne['y-tsne'] = tsne_results[:, 1]
df_tsne['outlier'] = (df_tsne['x-tsne'].pow(2) + df_tsne['y-tsne'].pow(2)).pow(0.5) >= 5

df_tsne.to_csv('tsne.tsv', sep='\t')
