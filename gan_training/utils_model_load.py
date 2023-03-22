
import torch
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
from collections import OrderedDict


def get_parameter_number(net, name= "un-named net"):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Name=', name, 'Total=', total_num, '# of Trainable Params.=', trainable_num, '# of Fixed Params.=', total_num-trainable_num)
    return trainable_num
    

def load_part_model(m_fix, m_ini):
    dict_fix = m_fix.state_dic()
    dict_ini = m_ini.state_dic()

    dict_fix = {k: v for k, v in dict_fix.items() if k in dict_ini and k.find('embedding')==-1 and k.find('fc') == -1}
    dict_ini.update(dict_fix)
    m_ini.load_state_dict(dict_ini)
    return m_ini


def load_weights_without_module(model, dict, strict=False):
    # functions: create new OrderedDict that does not contain `module` (from nn.DataParallel)
    new_state_dict = OrderedDict()
    for k, v in dict.items():
        if 'module' in k:
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    # load params
    model_dict = model.state_dict()
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict, strict=strict)
    return model


transform = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Lambda(lambda x: x + 1./128 * torch.rand(x.size())),
    ])


def get_parameter_num(model, task_id=0):
    p_num = 0

    p_num += model.AdaFM_fc.gamma.shape[1] + model.AdaFM_fc.beta.shape[1] + model.AdaFM_fc.b.shape[0]

    h1 = model.resnet_0_0.AdaFM_0.global_gamma_u[0].shape[0]
    w1 = model.resnet_0_0.AdaFM_0.global_num_gamma[task_id].cpu().data
    h2 = model.resnet_0_0.AdaFM_0.global_beta_u[0].shape[0]
    w2 = model.resnet_0_0.AdaFM_0.global_num_beta[task_id].cpu().data
    c = model.resnet_0_0.AdaFM_0.b.shape[0]
    p_num += 2*h1*w1 + w1 + 2*h2*w2 + w2 +c
    h1 = model.resnet_0_0.AdaFM_1.global_gamma_u[0].shape[0]
    w1 = model.resnet_0_0.AdaFM_1.global_num_gamma[task_id].cpu().data
    h2 = model.resnet_0_0.AdaFM_1.global_beta_u[0].shape[0]
    w2 = model.resnet_0_0.AdaFM_1.global_num_beta[task_id].cpu().data
    c = model.resnet_0_0.AdaFM_1.b.shape[0]
    p_num += 2*h1*w1 + w1 + 2*h2*w2 + w2 + c

    h1 = model.resnet_1_0.AdaFM_0.global_gamma_u[0].shape[0]
    w1 = model.resnet_1_0.AdaFM_0.global_num_gamma[task_id].cpu().data
    h2 = model.resnet_1_0.AdaFM_0.global_beta_u[0].shape[0]
    w2 = model.resnet_1_0.AdaFM_0.global_num_beta[task_id].cpu().data
    c = model.resnet_1_0.AdaFM_0.b.shape[0]
    p_num += 2*h1*w1 + w1 + 2*h2*w2 + w2 + c
    h1 = model.resnet_1_0.AdaFM_1.global_gamma_u[0].shape[0]
    w1 = model.resnet_1_0.AdaFM_1.global_num_gamma[task_id].cpu().data
    h2 = model.resnet_1_0.AdaFM_1.global_beta_u[0].shape[0]
    w2 = model.resnet_1_0.AdaFM_1.global_num_beta[task_id].cpu().data
    c = model.resnet_1_0.AdaFM_1.b.shape[0]
    p_num += 2*h1*w1 + w1 + 2*h2*w2 + w2 + c

    h1 = model.resnet_2_0.AdaFM_0.global_gamma_u[0].shape[0]
    w1 = model.resnet_2_0.AdaFM_0.global_num_gamma[task_id].cpu().data
    h2 = model.resnet_2_0.AdaFM_0.global_beta_u[0].shape[0]
    w2 = model.resnet_2_0.AdaFM_0.global_num_beta[task_id].cpu().data
    c = model.resnet_2_0.AdaFM_0.b.shape[0]
    p_num += 2*h1*w1 + w1 + 2*h2*w2 + w2 + c
    h1 = model.resnet_2_0.AdaFM_1.global_gamma_u[0].shape[0]
    w1 = model.resnet_2_0.AdaFM_1.global_num_gamma[task_id].cpu().data
    h2 = model.resnet_2_0.AdaFM_1.global_beta_u[0].shape[0]
    w2 = model.resnet_2_0.AdaFM_1.global_num_beta[task_id].cpu().data
    c = model.resnet_2_0.AdaFM_1.b.shape[0]
    p_num += 2*h1*w1 + w1 + 2*h2*w2 + w2 + c

    h1 = model.resnet_3_0.AdaFM_0.global_gamma_u[0].shape[0]
    w1 = model.resnet_3_0.AdaFM_0.global_num_gamma[task_id].cpu().data
    h2 = model.resnet_3_0.AdaFM_0.global_beta_u[0].shape[0]
    w2 = model.resnet_3_0.AdaFM_0.global_num_beta[task_id].cpu().data
    c = model.resnet_3_0.AdaFM_0.b.shape[0]
    p_num += 2*h1*w1 + w1 + 2*h2*w2 + w2 + c
    h1 = model.resnet_3_0.AdaFM_1.global_gamma_u[0].shape[0]
    w1 = model.resnet_3_0.AdaFM_1.global_num_gamma[task_id].cpu().data
    h2 = model.resnet_3_0.AdaFM_1.global_beta_u[0].shape[0]
    w2 = model.resnet_3_0.AdaFM_1.global_num_beta[task_id].cpu().data
    c = model.resnet_3_0.AdaFM_1.b.shape[0]
    p_num += 2*h1*w1 + w1 + 2*h2*w2 + w2 + c

    h1 = model.resnet_4_0.AdaFM_0.global_gamma_u[0].shape[0]
    w1 = model.resnet_4_0.AdaFM_0.global_num_gamma[task_id].cpu().data
    h2 = model.resnet_4_0.AdaFM_0.global_beta_u[0].shape[0]
    w2 = model.resnet_4_0.AdaFM_0.global_num_beta[task_id].cpu().data
    c = model.resnet_4_0.AdaFM_0.b.shape[0]
    p_num += 2*h1*w1 + w1 + 2*h2*w2 + w2 + c
    h1 = model.resnet_4_0.AdaFM_1.global_gamma_u[0].shape[0]
    w1 = model.resnet_4_0.AdaFM_1.global_num_gamma[task_id].cpu().data
    h2 = model.resnet_4_0.AdaFM_1.global_beta_u[0].shape[0]
    w2 = model.resnet_4_0.AdaFM_1.global_num_beta[task_id].cpu().data
    c = model.resnet_4_0.AdaFM_1.b.shape[0]
    p_num += 2*h1*w1 + w1 + 2*h2*w2 + w2 + c

    h1 = model.resnet_5_0.AdaFM_0.global_gamma_u[0].shape[0]
    w1 = model.resnet_5_0.AdaFM_0.global_num_gamma[task_id].cpu().data
    h2 = model.resnet_5_0.AdaFM_0.global_beta_u[0].shape[0]
    w2 = model.resnet_5_0.AdaFM_0.global_num_beta[task_id].cpu().data
    c = model.resnet_5_0.AdaFM_0.b.shape[0]
    p_num += 2*h1*w1 + w1 + 2*h2*w2 + w2 + c
    h1 = model.resnet_5_0.AdaFM_1.global_gamma_u[0].shape[0]
    w1 = model.resnet_5_0.AdaFM_1.global_num_gamma[task_id].cpu().data
    h2 = model.resnet_5_0.AdaFM_1.global_beta_u[0].shape[0]
    w2 = model.resnet_5_0.AdaFM_1.global_num_beta[task_id].cpu().data
    c = model.resnet_5_0.AdaFM_1.b.shape[0]
    p_num += 2*h1*w1 + w1 + 2*h2*w2 + w2 + c

    h1 = model.resnet_6_0.AdaFM_0.global_gamma_u[0].shape[0]
    w1 = model.resnet_6_0.AdaFM_0.global_num_gamma[task_id].cpu().data
    h2 = model.resnet_6_0.AdaFM_0.global_beta_u[0].shape[0]
    w2 = model.resnet_6_0.AdaFM_0.global_num_beta[task_id].cpu().data
    c = model.resnet_6_0.AdaFM_0.b.shape[0]
    p_num += 2*h1*w1 + w1 + 2*h2*w2 + w2 + c
    h1 = model.resnet_6_0.AdaFM_1.global_gamma_u[0].shape[0]
    w1 = model.resnet_6_0.AdaFM_1.global_num_gamma[task_id].cpu().data
    h2 = model.resnet_6_0.AdaFM_1.global_beta_u[0].shape[0]
    w2 = model.resnet_6_0.AdaFM_1.global_num_beta[task_id].cpu().data
    c = model.resnet_6_0.AdaFM_1.b.shape[0]
    p_num += 2*h1*w1 + w1 + 2*h2*w2 + w2 + c
    return p_num


def decompose_film_layer_g(model):
    # for generator only
    stdd = 1.0
    dict_all   = model.state_dict()
    model_dict = model.state_dict()

    for k, v in dict_all.items():
        # modulate fc layers
        if k.find(f'style') >= 0 and k.find(f'weight') >= 0:
            idx   = k[6]
            w_mu  = v.mean([1], keepdim=True)
            w_std = v.std([1], keepdim=True) * stdd
            dict_all[k].data = (v - w_mu)/(w_std)
            dict_all[f'film_layer.{idx}.gamma'].data = w_std.data.t()
            dict_all[f'film_layer.{idx}.beta'].data  = w_mu.data.t()
        
        # modulate conv layers
        elif k.find(f'convs') >= 0 and k.find(f'conv.weight') >= 0:
            idx   = k.find(f'conv.')
            w_mu  = v.mean([3,4], keepdim=True)
            w_std = v.std([3,4], keepdim=True) * stdd
            dict_all[k].data = (v - w_mu)/(w_std)
            dict_all[k[:idx]+'conv.style_gamma'].data = w_std.data
            dict_all[k[:idx]+'conv.style_beta'].data  = w_mu.data    
        
        # modulate to_RGB fc layers
        elif k.find(f'to_rgbs') >= 0 and k.find(f'modulation.weight') >= 0:
            idx   = k.find(f'conv.')
            w_mu  = v.mean([1], keepdim=True)
            w_std = v.std([1], keepdim=True) * stdd
            dict_all[k].data = (v - w_mu)/(w_std)
            dict_all[k[:idx]+'conv.film_layer.gamma'].data = w_std.data.t()
            dict_all[k[:idx]+'conv.film_layer.beta'].data  = w_mu.data.t()  
    model_dict.update(dict_all)
    model.load_state_dict(model_dict)
    return model


def decompose_film_layer_d(model):
    # for generator only
    stdd = 1.0
    dict_all   = model.state_dict()
    model_dict = model.state_dict()

    for k, v in dict_all.items():
        if k.find(f'style') >= 0 and k.find(f'weight') >= 0:
            idx = k[6]
            w_mu  = v.mean([1], keepdim=True)
            w_std = v.std([1], keepdim=True) * stdd
            dict_all[k].data = (v - w_mu)/(w_std)
            dict_all[f'film_layer.{idx}.gamma'].data = w_std.data.t()
            dict_all[f'film_layer.{idx}.beta'].data  = w_mu.data.t()
    model_dict.update(dict_all)
    model.load_state_dict(model_dict)
    return model
