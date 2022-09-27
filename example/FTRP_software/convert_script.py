import torch

device = torch.device("cuda:0")

# weight_path="/home/lab/ckq/S2ANet/runs/train/exp369/weights/best.pt"
# model = torch.load(weight_path, map_location=device)['model']
# model.float()


weight_path="model_test.pt"
model = torch.load(weight_path, map_location=device)

example = torch.rand(1, 3, 1024, 1024).to(device)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)
# traced_script_module = torch.jit.script(model, example)


# save model
save_name = weight_path.replace(".pt", "_script.pt")
traced_script_module.save(save_name)
