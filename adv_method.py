import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from tqdm import tqdm
import pretrainedmodels
import timm
from utils import load_state_dict_mute


# flip horizontal
def fliplr(img):
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long() # N x C x H x W
    inv_idx = inv_idx.to('cuda')
    img_flip = img.index_select(3, inv_idx)
    return img_flip


# extract an img's feature using the target model
def extract_feature(model, img, linear_num, batch_size):
    ff = torch.FloatTensor(batch_size, linear_num).zero_()
    ff = ff.to('cuda')
    for i in range(2):
        if i == 1:
            new_img = fliplr(img)
            break
        else:
            new_img = img
        outputs = model(new_img)
        ff += outputs
    fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
    ff = ff.div(fnorm.expand_as(ff))
    return ff


class Adversarial_Method:
    def __init__(self):
        pass
    

"""
FGSM (Fast Gradient Sign Method)
Paper: Explaining and Harnessing Adversarial Examples (Goodfellow, I.J. et al., arXiv:1412.6572)
  Tag: White-box, Targeted-attack, One-shot, Specific-perturbation
"""
class FGSM(Adversarial_Method):
    def __init__(self):
        pass
    
    def FGSM_attack(self, image, epsilon, data_grad):
        perturbed_image = image + epsilon * data_grad.sign()
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        return perturbed_image
    
    def loss_func(self, score, gallery_label, target_label):
        pass
        
    def get_attack_feature(self, model, test_loader, epsilon, linear_num, gallery_feature, gallery_label):
        pbar = tqdm()
        print("FGSM attack evaluating:")
        attacked_feature = torch.FloatTensor(len(test_loader.dataset), linear_num)
        start = 0
        # print(list(model.parameters()))
        for iter, data in enumerate(test_loader):
            # torch.cuda.empty_cache()
            # torch.cuda.empty_cache()
            # torch.cuda.empty_cache()
            img, label = data
            n, c, h, w = img.size()
            # pbar.update(n)
            img.requires_grad_()
            img = img.to('cuda')
            img.retain_grad()
            # # img_features.shape = [32, 512]
            img_features = extract_feature(model, img, linear_num, n)
            # # scores.shape = [32, 19732]
            scores = torch.matmul(img_features, gallery_feature.T)
            same_label = torch.IntTensor(n, scores.shape[1]).zero_()
            same_label = same_label.to("cuda")
            for i in range(n):
                same_label[i, :] = (torch.tensor(gallery_label) == int(label[i])).float()
            
            # calc_loss
            loss = torch.mul(scores, same_label).sum()
            loss.backward()
            tmp_grad = img.grad
            img.detach()
            perturb_imgs = self.FGSM_attack(img, epsilon, tmp_grad)
            perturb_feature = extract_feature(model, perturb_imgs, linear_num, n)
            end = start + perturb_feature.shape[0]
            attacked_feature[start:end, :] = perturb_feature
            print("start:", start, "end:", end, )
            start = end
            del img
            del img_features
            del same_label
            del scores
        return attacked_feature
    
    def get_attack_img(self, model, img, label, epsilon, linear_num, gallery_feature, gallery_label):
        n, c, h, w = img.size()
        # print("img.size()", img.size())
        img = img.to("cuda")
        img.requires_grad_()
        img.retain_grad()
        img_features = extract_feature(model, img, linear_num, n)
        scores = torch.matmul(img_features, gallery_feature.T)
        # print("label =", label)
        # same_label = (torch.tensor(gallery_label) == int(label.item())).float()
        same_label = (torch.tensor(gallery_label) == int(label)).float()
        same_label = same_label.to('cuda')
        # calc_loss
        loss = torch.mul(scores, same_label).sum()
        loss.backward()
        tmp_grad = img.grad
        img.detach()
        perturbs_img = self.FGSM_attack(img, epsilon, tmp_grad)
        return perturbs_img
    
                
if __name__ == '__main__':
    x = [3, 4, 3, 4, 4]
    y = (x == 4)
    print(y, type(y))