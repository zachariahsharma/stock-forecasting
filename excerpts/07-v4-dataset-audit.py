# Source: notebooks/TrainAlgoTraderGRUNormalized.ipynb · cell 12
# Original code filtering windows with too few distinct prices.
# Viewing excerpt; see original notebook for context.

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
def auditLoop(i, row):
    if len(np.unique(row.T[-1])) < 15:
        return i
    return np.nan
one = np.load('./numpyFinalSaves/x_train1.npy')
results = np.array(Parallel(n_jobs=-1)(delayed(auditLoop)(i, row) for i, row in tqdm(enumerate(one))))
results = results[~np.isnan(results)]
one = np.delete(one, results.astype('int'))
np.save('./numpyFinalSaves/x_train1Audit.npy', one)
del one
two = np.load('./numpyFinalSaves/y_train1.npy')
two = np.delete(two, results.astype('int'))
np.save('./numpyFinalSaves/y_train2Audit.npy', two)
