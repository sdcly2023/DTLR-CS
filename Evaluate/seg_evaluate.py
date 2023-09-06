import cv2
import segmentation_metrics as sm
import os
from tqdm import tqdm
from PIL import Image
import numpy as np


def binary_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')


if __name__ == "__main__":
    seg_path = r'D:\software_package\figure_seg\swimseg_binary\Ours'
    gd_path = r'D:\software_package\figure_seg\swimseg\testset\GT'
    seg = sorted(os.listdir(seg_path))
    precise = []
    recall = []
    f1 = []
    Jaccard_Similarity_Index = []
    Threshold_Jaccard_Index = []
    Dice = []
    Sensitivity = []
    Specificity = []
    Accuracy = []

    for name in tqdm(seg):
        mask = binary_loader(os.path.join(gd_path, name))
        mask_arr = np.asarray(mask, np.float32)
        mask_arr[mask_arr == 255] = 1

        predicted_mask = binary_loader(os.path.join(seg_path, name))
        predicted_mask_arr = np.asarray(predicted_mask, np.float32)
        predicted_mask_arr[predicted_mask_arr == 255] = 1

        metrics = sm.calculate(masks=[mask_arr], predicted_masks=[predicted_mask_arr], jaccard_threshold=0.65)

        precise.append(metrics["precise"])
        recall.append(metrics['recall'])
        f1.append(metrics['F1'])
        Jaccard_Similarity_Index.append(metrics['jaccard_similarity_index_(iou_score)'])
        Threshold_Jaccard_Index.append(metrics['threshold_jaccard_index'])
        Dice.append(metrics['dice_coefficient'])
        Sensitivity.append(metrics['sensitivity'])
        Specificity.append(metrics['specificity'])
        Accuracy.append(metrics['accuracy'])

    precise_per = np.sum(precise) / len(seg)
    recall_per = np.sum(recall) / len(seg)
    f1_per = np.sum(f1) / len(seg)
    Jaccard_Similarity_Index_per = np.sum(Jaccard_Similarity_Index) / len(seg)
    Threshold_Jaccard_Index_per = np.sum(Threshold_Jaccard_Index) / len(seg)
    Dice_per = np.sum(Dice) / len(seg)
    Sensitivity_per = np.sum(Sensitivity) / len(seg)
    Specificity_per = np.sum(Specificity) / len(seg)
    Accuracy_per = np.sum(Accuracy) / len(seg)

    print(" precise_per = {}".format(precise_per))
    print(" recall_per = {}".format(recall_per))
    print(" f1_per = {}".format(f1_per))
    print(" Jaccard_Similarity_Index_per = {}".format(Jaccard_Similarity_Index_per))
    print(" Threshold_Jaccard_Index_per = {}".format(Threshold_Jaccard_Index_per))
    print(" Dice_per = {}".format(Dice_per))
    print(" Sensitivity_per = {}".format(Sensitivity_per))
    print(" Specificity_per = {}".format(Specificity_per))
    print(" Accuracy_per = {}".format(Accuracy_per))

    file = open(r"D:\software_package\figure_seg\swimseg_binary" + seg_path.split("binary")[-1] + "_seg_evaluate.txt", 'w')
    file.write(" precise_per = {}".format(precise_per) +
               " recall_per = {}".format(recall_per) +
               " f1_per = {}".format(f1_per) +
               " Jaccard_Similarity_Index_per = {}".format(Jaccard_Similarity_Index_per) +
               " Threshold_Jaccard_Index_per = {}".format(Threshold_Jaccard_Index_per) +
               " Dice_per = {}".format(Dice_per) +
               " Sensitivity_per = {}".format(Sensitivity_per) +
               " Specificity_per = {}".format(Specificity_per) +
               " Accuracy_per = {}".format(Accuracy_per)
               )
    file.close()
