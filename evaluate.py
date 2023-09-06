import cv2
from medpy.metric import binary
import os 
import numpy as np
import torch

def get_binary(img):
    """获取连通域

    :param img: 输入图片
    :return: 最大连通域
    """
    # 灰度化, 二值化, 连通域分析
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, img_bin = cv2.threshold(img_gray, 127, 1.0, cv2.THRESH_BINARY)
    
    return img_bin

def dice_coef(output, target):#output为预测结果 target为真实结果
    smooth = 1e-5 #防止0除

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()


    return 2 * np.sum(output * target) / (np.sum(output) + np.sum(target) + smooth)

        
def iou_score(output, target):
    smooth = 1e-5
 
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
 
    return (intersection + smooth) / (union + smooth)


def main():
    pred_dir = '/data/pulianghao/UCTransNet-ceil-main/MoNuSeg_visualize_test'
    GT_dir = '/data/pulianghao/UCTransNet-ceil-main/datasets/MoNuSeg/Test_Folder/labelcol'
    
    pred_img_paths = sorted(os.listdir(pred_dir))

    dices = 0.0
    hd95s = 0.0
    hds = 0.0
    jcs = 0.0
    Precisions = 0.0
    ious = 0.0
    index = 0
    for i in range(len(pred_img_paths)):
        img_path = os.path.join(pred_dir,pred_img_paths[i])
        gt_path = os.path.join(GT_dir,pred_img_paths[i].split('_predict')[0]+'.png')
        # 1.导入图片
        img_cs1 = cv2.imread(img_path)
        img_cs2 = cv2.imread(gt_path)
        # 2.获取图片连通域
        cnt_cs1 = get_binary(img_cs1)
        cnt_cs2 = get_binary(img_cs2)
        #iou
        iou = iou_score(cnt_cs1, cnt_cs2)
        ious+=iou
        #dice
        dice=binary.dc(cnt_cs1, cnt_cs2)
        # dice = dice_coef(cnt_cs1, cnt_cs2)
        dices+=dice
        #hd95
        hd95=binary.hd95(cnt_cs1, cnt_cs2, voxelspacing=None)
        hd95s+=hd95
        #hd
        hd=binary.hd(cnt_cs1, cnt_cs2, voxelspacing=None)
        hds+=hd
        #Jc
        jaccard=binary.jc(cnt_cs1, cnt_cs2)
        jcs+=jaccard
        # 计算精确度/阳性预测值
        Precision=binary.precision(cnt_cs1, cnt_cs2)
        Precisions+=Precision
        index = index+1
    print("交并比IoU = ",ious/index)
    print('Dice相似系数 = ',dices/index)
    print('95%豪斯多夫距离 = ',hd95s/index)
    print('豪斯多夫距离 = ',hds/index)
    print("Jaccard相似系数 = ",jcs/index)
    print("精确度/阳性预测值为 = ",Precisions/index)


if __name__ == '__main__':
    main()

