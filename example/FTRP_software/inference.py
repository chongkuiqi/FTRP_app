import torch
import cv2
import numpy as np

from  utils.roi_align_rotated.functions.roi_align_rotated import get_roi_feat

def preprocess(img):
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    # ascontiguousarray函数将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快。
    img = np.ascontiguousarray(img)

    # 图像数据转化为tensor，并放入设备中
    img = torch.from_numpy(img)
    img = img.float()  # uint8 to fp16/32
    if img.ndimension() == 3:
        # 增加一个维度，其实就是batch维度
        img = img.unsqueeze(0)

    img = img_batch_normalize(img)

    return img

def img_batch_normalize(img):
    # img_normalize_mean =  np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(3,1,1)
    # img_normalize_std = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(3,1,1)
    # img = (img - img_normalize_mean) / img_normalize_std

    img_normalize_mean =  [123.675, 116.28, 103.53]
    img_normalize_std = [58.395, 57.12, 57.375]
    
    img[:,0,:,:] = (img[:,0,:,:] - img_normalize_mean[0]) / img_normalize_std[0]
    img[:,1,:,:] = (img[:,1,:,:] - img_normalize_mean[1]) / img_normalize_std[1]
    img[:,2,:,:] = (img[:,2,:,:] - img_normalize_mean[2]) / img_normalize_std[2]

    return img

# device = torch.device("cpu")
device = torch.device("cuda:0")

img_path = "vedai_1.png"
# img_path = "vedai_2.png"

img = cv2.imread(img_path)
img = preprocess(img)
img = img.to(device)


model_path = "./exp361_no_align_cuda_script.pt"
model = torch.jit.load(model_path).to(device)
# model.evel()
with torch.no_grad():
    outputs = model.forward(img)

print(len(outputs))

print(img.device)
roi_1 = torch.tensor((0.0000,  443.0000,  614.0000 , 166.0000 ,  19.0000,   -0.3122), device=device).reshape(-1,6)

roi_2 = torch.tensor((0.0000 , 988.0000  ,190.0000 ,  50.0000,    9.0000 ,   1.3306), device=device).reshape(-1,6)
# roi_2 = torch.tensor((0.0000 , 988.0000  ,190.0000 ,  50.0000,    9.0000 ,   -0.03122), device=device).reshape(-1,6)


featmap_strides = [8,16,32]
level_id = 2
print(outputs[level_id].reshape(-1)[-20:])
roi_1_fe = get_roi_feat(outputs[level_id], roi_1, spatial_scale=1/featmap_strides[level_id])
roi_2_fe = get_roi_feat(outputs[level_id], roi_2, spatial_scale=1/featmap_strides[level_id])


diff = torch.abs(roi_1_fe-roi_2_fe).sum().item()

print(diff)