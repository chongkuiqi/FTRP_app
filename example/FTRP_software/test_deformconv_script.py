import torch
import cv2
from models.alignconv import AlignConv

# device = torch.device("cuda:0")
device = torch.device("cpu")

model = AlignConv(in_channels=256, out_channels=256, kernel_size=3,)
model.to(device)
model.eval()

torch.manual_seed(0)
example = torch.rand(1, 256, 32, 32).to(device)

# anchors shape [B,H,W,5]
anchors = torch.rand((1, 32, 32, 5)).to(device) * 1024
stride = torch.tensor(8)

traced_script_module = torch.jit.trace(model.eval(), (example, anchors, stride))
# save model
save_name = "align_script.pt"
traced_script_module.save(save_name)

# with torch.no_grad():
#     # outputs = model(example, anchors, stride)
#     traced_script_module = torch.jit.trace(model.eval(), (example, anchors, stride))
#     # save model
#     save_name = "align_script.pt"
#     traced_script_module.save(save_name)


# model_path = "/home/ckq/software/example/FTRP_software/model_no_align_script.pt"
# model = torch.jit.load(model_path).to(device)
# outputs = model.forward(example)

# model_path = "/home/ckq/software/example/FTRP_software/model_no_align.pt"
# model = torch.load(model_path).to(device)
# outputs = model(example)

