import torch
from models.retinanet3 import RetinaNet_extract as Model
from utils.general import intersect_dicts
from models.backbone import load_state_dict
import cv2
import numpy as np

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

device = torch.device("cpu")
# device = torch.device("cuda:0")
img_path = "./vedai_1.png"


img = cv2.imread(img_path)
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
img = img.to(device)



# print(model)
# weight_path = 'exp383.pt'
weight_path = 'exp382.pt'
pretrained = torch.load(weight_path, map_location=device)['model']
model = Model().to(device)
model = load_state_dict(model, pretrained.state_dict())
model.float()
model.eval()

# print(model(img))
with torch.no_grad():
    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(model, img)
    # traced_script_module = torch.jit.script(model, img)

# save model
save_name = weight_path.replace(".pt", "_fe_script.pt")
traced_script_module.save(save_name)


