import torch
import cv2


device = torch.device("cuda:0")
example = torch.rand(1, 3, 1024, 1024).to(device)
img_path = "/home/ckq/software/example/FTRP_software/vedai_1.png"


model_path = "/home/ckq/software/example/FTRP_software/model_no_align_script.pt"
model = torch.jit.load(model_path).to(device)
outputs = model.forward(example)

# model_path = "/home/ckq/software/example/FTRP_software/model_no_align.pt"
# model = torch.load(model_path).to(device)
# outputs = model(example)

