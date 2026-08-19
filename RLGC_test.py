encoding="ISO-8859-1"
import torch
import os
from torch import nn, optim
from torchvision import datasets, transforms
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
from faceUtil.face_utils import FaceDetector, norm_crop
from model.DGNetFF import DGNet as Network
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.CenterCrop((250, 200)),
    # transforms.Resize((100, 100)),
    transforms.ToTensor(),
])

if __name__ == "__main__":

    modelpath = '/home/xukaiwen/RLGC算法/RLGC算法/RLGC测试代码/model_weight/RLGC.pth'

    # imgpath = '/home/xukaiwen01/Deepfake_Hu/test_image/OCVAE/032-66-real.jpg'
    imgpath = '/home/xukaiwen/RLGC算法/RLGC算法/RLGC测试图片/fake/2.jpg'

    # 读取图片，用人脸检测器分割出人脸--start
    img = cv2.imread(imgpath)
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img1)
    plt.axis('off')  # 关掉坐标轴为 off
    plt.title('原图', fontproperties='SimHei',fontsize=20)
    plt.show()
    plt.close()

    face_detector = FaceDetector()
    face_detector.load_checkpoint("faceUtil/pre_model_weight/RetinaFace-Resnet50-fixed.pth")

    boxes, landms = face_detector.detect(img)
    if boxes.shape[0] == 0:
        print('未检测出人脸，任务结束。')
        sys.exit()
    areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    max_face_idx = areas.argmax()
    landm = landms[max_face_idx]
    landmarks = landm.detach().numpy().reshape(5, 2).astype(int)
    img = norm_crop(img, landmarks, image_size=320)
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img2)
    plt.axis('off')  # 关掉坐标轴为 off
    plt.title('脸部区域提取',fontproperties='SimHei',fontsize=20)  # 图像题目
    plt.show()
    # 读取图片，用人脸检测器分割出人脸--end

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image = transform_test(img2)
    # plt.imshow(image)
    # plt.axis('off')  # 关掉坐标轴为 off
    # plt.title('修改后', fontproperties='SimHei', fontsize=20)
    # plt.show()
    # plt.close()
    image = image.float().to(device)
    image = image.unsqueeze(0)
    fakedect = Network(channel=64, arc='EfficientNet-B4', M=[8, 8, 8], N=[4, 8, 16]).cuda()
    fakedect.load_state_dict(torch.load(modelpath))
    fakedect.eval()
    with torch.no_grad():
        img_np = np.array(img)
        # Convert to grayscale
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        # Perform Canny edge detection
        gradient_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
        # Combine x and y gradients into a single gradient image
        gradient_image = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
        gradient_image = torch.from_numpy(gradient_image).to(device)
        Y_pre, recons_x, pg = fakedect(image,gradient_image)
        Y_pre = Y_pre.squeeze()
        pred_label = torch.argmax(Y_pre, dim=-1)

        # 将索引转换为0或1结果
        pred_result = pred_label.item()
if pred_result == 0:
    # 可视化recons_x和pg
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(recons_x.cpu().detach().numpy().squeeze().transpose(1, 2, 0))
    ax1.axis('off')
    ax1.set_title('虚假人脸空间重建', fontproperties='SimHei', fontsize=15)

    ax2.imshow(pg.cpu().detach().numpy().squeeze(), cmap='gray')
    ax2.axis('off')
    ax2.set_title('梯度特征重建', fontproperties='SimHei', fontsize=15)

    plt.suptitle('图像最终预测为：假', fontproperties='SimHei', fontsize=20)
    plt.show()
else:
    # 可视化recons_x和pg
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(recons_x.cpu().detach().numpy().squeeze().transpose(1, 2, 0))
    ax1.axis('off')
    ax1.set_title('真实人脸空间重建', fontproperties='SimHei', fontsize=15)

    ax2.imshow(pg.cpu().detach().numpy().squeeze(), cmap='gray')
    ax2.axis('off')
    ax2.set_title('梯度特征重建', fontproperties='SimHei', fontsize=15)

    plt.suptitle('图像最终预测为：真', fontproperties='SimHei', fontsize=20)
    plt.show()

