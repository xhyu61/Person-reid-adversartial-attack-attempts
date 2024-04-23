import torch
import os
from model import ft_net
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms

def target_id():
    return "0034"


def get_cos_sim(feature1, feature2):
    return F.cosine_similarity(feature1, feature2)


def jpg_to_tensor(jpg_path):
    with Image.open(jpg_path) as jpg:
        trans = transforms.Compose([
            transforms.ToTensor()
        ])
        img_tensor = trans(jpg).unsqueeze(dim=0)
        return img_tensor


def reid_loss(adv_tensor, adv_over_dif=10.0):
    device = "cuda"
    # load model
    model_path = "reid_model/model/ft_ResNet50/net_last.pth"
    model = ft_net()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    del checkpoint
    
    # get adv_tensor
    adv_tensor = adv_tensor.to(torch.float16)
    # print("adv_tensor:", adv_tensor.unsqueeze(0).shape)
    adv_feature = model(adv_tensor.unsqueeze(0))
        
    # calc similarity
    adv_loss = dif_sim = adv_sim = torch.tensor([0.]).to(device)
    adv_id = target_id()
    gallery_path = "Market/pytorch/small_query"
    gallery_ids = os.listdir(gallery_path)
    for id in gallery_ids:
        id_path = gallery_path + "/" + id
        id_imgs = os.listdir(id_path)
        for img in id_imgs:
            img_tensor = jpg_to_tensor(id_path + "/" + img).to(device)
            img_feature = model(img_tensor)
            cos_sim = torch.abs(get_cos_sim(img_feature, adv_feature))
            # print("cos_sim:", cos_sim)
            del img_tensor
            del img_feature
            if id == adv_id:
                adv_sim = adv_sim + cos_sim
            else:
                dif_sim = dif_sim + cos_sim
    adv_loss = adv_loss + adv_over_dif * adv_sim + dif_sim
    del adv_sim
    del dif_sim
    del cos_sim
    del model
    # print("adv_loss.shape", adv_loss.shape)
    return adv_loss
    

def calc_acc():
    raise NotImplementedError