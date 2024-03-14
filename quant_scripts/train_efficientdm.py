'''
For training, remember to set self.total_steps = 100 in quant_scripts/quant_layer.py/TemporalActivationQuantizer
'''
import sys, time, datetime
sys.path.append(".")
sys.path.append('./taming-transformers')
from taming.models import vqgan

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
torch.cuda.manual_seed(3407)
import torch.nn as nn
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler, DDIMSampler_trainer
import ldm.globalvar as globalvar

import numpy as np 
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid


from quant_scripts.quant_model import QuantModel_intnlora
from quant_scripts.quant_layer import QuantModule_intnlora, SimpleDequantizer

n_bits_w = 4
n_bits_a = 4

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        # print(_, ':', param.numel())
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )

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
    n_samples_per_class = 2
    ## Quality, sampling speed and diversity are best controlled via the `scale`, `ddim_steps` and `ddim_eta` variables
    ddim_steps = 100
    ddim_eta = 1.0
    scale = 1.5   # for  guidance

    fp_model = get_model()
    fp_model.cond_stage_model.cpu()
    fp_model.first_stage_model.cpu()
    model = get_model()
    # model.first_stage_model.cpu()
    dmodel = model.model.diffusion_model
    dmodel.cuda()
    dmodel.eval()
    from quant_scripts.quant_dataset import DiffusionInputDataset
    from torch.utils.data import DataLoader

    dataset = DiffusionInputDataset('/home/hyf/latent-diffusion/DiffusionInput_250steps.pth')
    data_loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True) ## each sample is (16,4,32,32)
    
    wq_params = {'n_bits': n_bits_w, 'channel_wise': True, 'scale_method': 'mse'}
    aq_params = {'n_bits': n_bits_a, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': True}
    qnn = QuantModel_intnlora(model=dmodel, weight_quant_params=wq_params, act_quant_params=aq_params, num_steps=ddim_steps)
    qnn.cuda()
    qnn.eval()

    print('Setting the first and the last layer to 8-bit')
    qnn.set_first_last_layer_to_8bit()

    cali_images, cali_t, cali_y = get_train_samples(data_loader, num_samples=1024)
    device = next(qnn.parameters()).device
    # Initialize weight quantization parameters
    qnn.set_quant_state(True, True)

    for name, module in qnn.named_modules():
        if isinstance(module, QuantModule_intnlora) and module.ignore_reconstruction is False:
            module.intn_dequantizer = SimpleDequantizer(uaq=module.weight_quantizer, weight=module.weight)

    ckpt = torch.load('quantw4a4_naiveQ_intsaved.pth'.format(n_bits_w), map_location='cpu')
    qnn.load_state_dict(ckpt, strict=False) ## no lora weight in ckpt

    for name, module in qnn.named_modules():
        if isinstance(module, QuantModule_intnlora) and module.ignore_reconstruction is False:
            module.weight.data = module.weight.data.byte()
            module.intn_dequantizer.delta.data = module.weight_quantizer.delta ## share the same delta and zp
            module.intn_dequantizer.zero_point.data = module.weight_quantizer.zero_point
    
    print('First run to init model...') ## need run to init temporal act quantizer
    with torch.no_grad():
        _ = qnn(cali_images[:4].to(device),cali_t[:4].to(device),cali_y[:4].to(device))

    setattr(model.model, 'diffusion_model', qnn)

    for name, param in model.named_parameters():
        if 'lora' in name or 'delta' in name or 'zp_list' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    for name, param in fp_model.named_parameters():
        param.requires_grad = False

    print_trainable_parameters(model)
    
    avg_delta_list = []
    from transformers import get_linear_schedule_with_warmup
    NUM_EPOCHS = 160
    firstone = True
    for name, module in model.named_modules():
        if isinstance(module, QuantModule_intnlora):
            if len(module.act_quantizer.delta_list) != ddim_steps:
                raise ValueError('Wrong act_quantizer.delta_list length')
            avg_delta = torch.sum(module.weight_quantizer.delta) / torch.numel(module.weight_quantizer.delta)

            params = [param for name, param in module.named_parameters() if 'lora' in name]
            if firstone:
                optimizer = torch.optim.AdamW(params, lr=avg_delta / 2500, foreach=False)
                firstone = False
            else:
                optimizer.add_param_group({'params': params, 'lr': avg_delta / 2500})

            params = [param for name, param in module.named_parameters() if 'delta' in name and 'list' not in name] ## weight quantizer
            if firstone:
                optimizer = torch.optim.AdamW(params, lr=1e-6, foreach=False)
                firstone = False
            else:
                optimizer.add_param_group({'params': params, 'lr': 1e-6})

            params = [param for name, param in module.named_parameters() if 'delta_list' in name or 'zp_list' in name] ## act quantizer
            if firstone:
                optimizer = torch.optim.AdamW(params, lr=5e-4, foreach=False)
                firstone = False
            else:
                optimizer.add_param_group({'params': params, 'lr': 5e-4})

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(NUM_EPOCHS * ddim_steps),
    )

    # Saving optimizer state for each step, which has little effect on performance. Uncomment line 368-370 & 405 in ddim.py if use it.
    # globalvar.init_state_list(ddim_steps) 

    model.eval()
    sampler = DDIMSampler_trainer(fp_model, quant_model=model, lr_scheduler=lr_scheduler, optimizer=optimizer)
    eval_sampler = DDIMSampler(model)
    eval_out_dir = os.path.join('experiments_log', str(datetime.datetime.now()))
    os.mkdir(eval_out_dir)
    all_samples = list()

    with model.ema_scope():
        for epoch in range(NUM_EPOCHS):
            print(f'{epoch=}')
            torch.cuda.manual_seed(3407+epoch)
            uc = model.get_learned_conditioning(
                {model.cond_stage_key: torch.tensor(n_samples_per_class*[1000]).to(model.device)}
                )
            t0 = time.time()

            class_labels = torch.randint(low=0,
                                        high=1000,
                                        size=(n_samples_per_class,),
                                        device=device)
            xc = torch.tensor(class_labels).to(model.device)
            c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                            conditioning=c,
                                            batch_size=n_samples_per_class,
                                            shape=[3, 64, 64],
                                            verbose=False,
                                            unconditional_guidance_scale=scale,
                                            unconditional_conditioning=uc, 
                                            eta=ddim_eta)

            
            # generate samples for visual evaluation
            if epoch % 16 == 0:
                print('Evaluating...')
                torch.cuda.manual_seed(3407)
                model.eval()
                eval_classes = [88, 417]
                uc = model.get_learned_conditioning(
                    {model.cond_stage_key: torch.tensor(n_samples_per_class*[1000]).to(model.device)}
                    )
                
                for class_label in eval_classes:
                    t0 = time.time()
                    print(f"rendering {n_samples_per_class} examples of class '{class_label}' in {ddim_steps} steps and using s={scale:.2f}.")
                    xc = torch.tensor(n_samples_per_class*[class_label])
                    c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
                    
                    samples_ddim, _ = eval_sampler.sample(S=ddim_steps,
                                                    conditioning=c,
                                                    batch_size=n_samples_per_class,
                                                    shape=[3, 64, 64],
                                                    verbose=False,
                                                    unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=uc, 
                                                    eta=ddim_eta)

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
                                                min=0.0, max=1.0)
                    all_samples.append(x_samples_ddim)

                    t1 = time.time()
                    print('throughput : {}'.format(x_samples_ddim.shape[0] / (t1 - t0)))
                # display as grid
                grid = torch.stack(all_samples, 0)
                grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                grid = make_grid(grid, nrow=n_samples_per_class)

                # to image
                grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                image_to_save = Image.fromarray(grid.astype(np.uint8))
                image_to_save.save(os.path.join(eval_out_dir, "epoch{}.jpg".format(epoch)))
                all_samples.clear()

    torch.save(model.state_dict(), 'quantw{}a{}_{}steps_efficientdm.pth'.format(n_bits_w, n_bits_a, ddim_steps))

