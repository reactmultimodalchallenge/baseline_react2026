import numpy as np
from tslearn.metrics import dtw
from functools import partial
import multiprocessing as mp
from tqdm import tqdm


def _func(target, pred):
    target = target.numpy()
    pred = pred.numpy()

    # num_preds = pred.shape[0]
    min_dwt_sum = 0
    for i in range(pred.shape[0]):
        dwt_list = []
        for j in range(target.shape[0]):
            emotion = target[j]
            res = 0
            for st, ed, weight in [(0, 15, 1 / 15), (15, 17, 1), (17, 25, 1 / 8)]:
                res += weight * dtw(pred[i].astype(np.float32)[:, st: ed], emotion.astype(np.float32)[:, st: ed])
            dwt_list.append(res)
        min_dwt_sum += min(dwt_list)
    return min_dwt_sum


def _func_star(args):
    target, pred = args
    return _func(target, pred)


def compute_FRD(preds, targets, p=1):
    tasks = list(zip(targets, preds))
    FRD_list = []

    with mp.Pool(processes=p) as pool:
        for result in tqdm(
            pool.imap_unordered(_func_star, tasks),
            total=len(tasks),
            desc="Computing FRD"
        ):
            FRD_list.append(result)

    return np.mean(FRD_list)


