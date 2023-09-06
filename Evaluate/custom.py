import os
from tqdm import tqdm
import numpy as np
from PIL import Image


def cohen_kappa(a: np.int32, b: np.int32) -> float:
    assert a.shape == b.shape
    po = (a == b).astype(np.float32).mean()
    categories = sorted(set(list(np.concatenate((a, b), axis=0))))
    mp = {}
    for i, c in enumerate(categories):
        mp[c] = i
    k = len(mp)
    sa = np.zeros(shape=(k,), dtype=np.int32)
    sb = np.zeros(shape=(k,), dtype=np.int32)
    n = a.shape[0]
    for x, y in zip(list(a), list(b)):
        sa[mp[x]] += 1
        sb[mp[y]] += 1
    pe = 0
    for i in range(k):
        pe += (sa[i] / n) * (sb[i] / n)
    kappa = (po - pe) / (1.0 - pe)
    return kappa


def binary_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')


if __name__ == "__main__":
    seg_path = r'D:\software_package\figure_seg\swimseg_binary\Unet'
    gd_path = r'D:\software_package\figure_seg\swimseg\testset\GT'
    seg = sorted(os.listdir(seg_path))
    kappa = []

    for name in tqdm(seg):
        mask = binary_loader(os.path.join(gd_path, name))
        mask_arr = np.asarray(mask, np.float32)
        mask_arr[mask_arr == 255] = 1
        mask_arr = mask_arr.flatten()

        predicted_mask = binary_loader(os.path.join(seg_path, name))
        predicted_mask_arr = np.asarray(predicted_mask, np.float32)
        predicted_mask_arr[predicted_mask_arr == 255] = 1
        predicted_mask_arr = predicted_mask_arr.flatten()

        kappa.append(cohen_kappa(mask_arr,predicted_mask_arr))

    kappa_per = np.sum(kappa) / len(seg)

    print(" Accuracy_per = {}".format(kappa_per))

    file = open(r"D:\software_package\figure_seg\swimseg_binary" + seg_path.split("binary")[-1] + "_custom.txt", 'w')
    file.write(" kappa = {}".format(kappa_per))
    file.close()
