import torch
ckpt = torch.load('quantw4a4_100steps_efficientdm.pth', map_location='cpu')
newsd = {}
for k, v in ckpt.items():
    if 'delta_list' in k or 'zp_list' in k:
        # newv = v[::12][:20]
        newv = v[::5]
    
    else:
        newv = v
    
    newsd[k] = newv

torch.save(newsd, 'quantw4a4_20steps_efficientdm.pth')