import torch

from unet3d_model.vnet3d import VNet
device = torch.device('cpu')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = VNet(in_channels=1, classes=2).to(device=device)

ckpt = torch.load('model_save/bone_best_model_vnet_900.tar', map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])
del ckpt
model.eval()
print(type(model))
# inputs = torch.rand(1,1,144,96,96).to(device)
inputs = torch.rand(1,1,80,160,160).to(device)
traced_script_module = torch.jit.trace(model, inputs)
traced_script_module.save('model_save/'+"boneCBCT_{}.pt".format(device))

