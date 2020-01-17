from utils import process
import numpy as np

_, _, labels, _, _, _ = process.load_data('cora')
stat = np.sum(labels, axis = 0)
print('cora: ', stat)
_, _, labels, _, _, _ = process.load_data('pubmed')
stat = np.sum(labels, axis = 0)
print('pubmed: ', stat)
_, _, labels, _, _, _ = process.load_data('citeseer')
stat = np.sum(labels, axis = 0)
print('citeseer: ', stat)

