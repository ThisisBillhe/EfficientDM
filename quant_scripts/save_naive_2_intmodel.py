import sys, time
sys.path.append(".")
sys.path.append('./taming-transformers')

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
torch.cuda.manual_seed(3407)
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config

import numpy as np 

from quant_scripts.quant_model import QuantModel
from quant_scripts.quant_layer import QuantModule

n_bits_w = 4
n_bits_a = 4

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def get_model():
    config = OmegaConf.load("configs/latent-diffusion/cin256-v2.yaml")  
    model = load_model_from_config(config, "models/ldm/cin256-v2/model.ckpt")
    return model

def get_train_samples(train_loader, num_samples):
    image_data, t_data, y_data = [], [], []
    for (image, t, y) in train_loader:
        image_data.append(image)
        t_data.append(t)
        y_data.append(y)
        if len(image_data) >= num_samples:
            break
    return torch.cat(image_data, dim=0)[:num_samples], torch.cat(t_data, dim=0)[:num_samples], torch.cat(y_data, dim=0)[:num_samples]

if __name__ == '__main__':
    model = get_model()
    dmodel = model.model.diffusion_model
    dmodel.cuda()
    dmodel.eval()
    
    wq_params = {'n_bits': n_bits_w, 'channel_wise': True, 'scale_method': 'max'}
    aq_params = {'n_bits': n_bits_a, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': True}
    qnn = QuantModel(model=dmodel, weight_quant_params=wq_params, act_quant_params=aq_params, need_init=False)
    qnn.cuda()
    qnn.eval()

    print('Setting the first and the last layer to 8-bit')
    qnn.set_first_last_layer_to_8bit()

    # cali_images, cali_t, cali_y = get_train_samples(data_loader, num_samples=1024)
    device = next(qnn.parameters()).device
    # Initialize weight quantization parameters
    qnn.set_quant_state(True, True)

    ckpt = torch.load('quantw4a4_naiveQ.pth'.format(n_bits_w), map_location='cpu')
    qnn.load_state_dict(ckpt)
    qnn.cuda()
    qnn.eval()

    for name, param in qnn.named_parameters():
        param.requires_grad = False

    for name, module in qnn.named_modules():
        if isinstance(module, QuantModule) and module.ignore_reconstruction is False: 
            x = module.weight

            x_int = torch.round(x / module.weight_quantizer.delta) + module.weight_quantizer.zero_point
            x_quant = torch.clamp(x_int, 0, module.weight_quantizer.n_levels - 1) 

            ## pack to int
            ori_shape = x_quant.shape
            if module.fwd_func is F.conv2d:
                x_quant = x_quant.flatten(1)
            i = 0
            row = 0
            intweight = x_quant.int().cpu().numpy().astype(np.uint8)
            qweight = np.zeros(
                (intweight.shape[0] // 8 * n_bits_w, intweight.shape[1]), dtype=np.uint8
            )
            while row < qweight.shape[0]:
                if n_bits_w in [2, 4, 8]:
                    for j in range(i, i + (8 // n_bits_w)):
                        qweight[row] |= intweight[j] << (n_bits_w * (j - i))
                    i += 8 // n_bits_w
                    row += 1      

            qweight = torch.tensor(qweight).cuda()
            qweight = qweight.reshape([qweight.shape[0]]+list(ori_shape[1:]))

            module.weight.data = qweight
    
    qnn_sd = qnn.state_dict()
    torch.save(qnn_sd, 'quantw{}a{}_naiveQ_intsaved.pth'.format(n_bits_w, n_bits_a))
    