import argparse
import math
import numpy as np
import os
import time
import torch
import torch.nn as nn
import yaml
from adv_method import FGSM
from model import ft_net
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from tqdm import tqdm
from utils import fuse_all_conv_bn


# ------------------------------------------------------------------
# Options
parser = argparse.ArgumentParser(description="Adversarial Attack")
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir',default='../Market/pytorch',type=str, help='./test_data')
parser.add_argument('--batchsize', default=32, type=int, help='batch size')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='name of the attacked target model')
parser.add_argument('--ibn', action='store_true', help='use ibn.' )
parser.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
parser.add_argument('--epsilon', default='0.1', type=float, help='strength of the attack')
parser.add_argument('--use_FGSM', action='store_true', help='use FGSM')
opt = parser.parse_args()

# ------------------------------------------------------------------
# Load config
config_path = os.path.join('./model', opt.name, 'opts.yaml')
with open(config_path, 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)

opt.stride = config['stride']

if 'nclasses' in config: # tp compatible with old config files
    opt.nclasses = config['nclasses']
else: 
    opt.nclasses = 751 # Market
    
if 'linear_num' in config:
    opt.linear_num = config['linear_num']

# ------------------------------------------------------------------
# prepare dataloaders
h, w = 256, 128 # Market set

data_transforms = transforms.Compose([
    transforms.Resize((h, w), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

str_ms = opt.ms.split(',')
ms = []
for s in str_ms:
    s_f = float(s)
    ms.append(math.sqrt(s_f))

data_dir = opt.test_dir

image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                              shuffle=False, num_workers=16) for x in ['gallery','query']}
test_loaders = torch.utils.data.DataLoader(image_datasets['query'], batch_size=opt.batchsize, 
                                           shuffle=False, num_workers=16)

class_names = image_datasets['query'].classes
use_gpu = torch.cuda.is_available()

# ------------------------------------------------------------------
# Load model
def load_network(network):
    save_path = os.path.join('./model',opt.name,'net_%s.pth'%opt.which_epoch)
    try:
        network.load_state_dict(torch.load(save_path))
    except: 
        if torch.cuda.get_device_capability()[0]>6 and len(opt.gpu_ids)==1 and int(version[0])>1: # should be >=7
            print("Compiling model...")
            # https://huggingface.co/docs/diffusers/main/en/optimization/torch2.0
            torch.set_float32_matmul_precision('high')
            network = torch.compile(network, mode="default", dynamic=True) # pytorch 2.0
        network.load_state_dict(torch.load(save_path))

    return network

# ------------------------------------------------------------------
# Extract feature
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    inv_idx = inv_idx.to('cuda')
    img = img.to('cuda')
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model,dataloaders):
    #features = torch.FloatTensor()
    count = 0
    pbar = tqdm()
    if opt.linear_num <= 0:
        if opt.use_swin or opt.use_swinv2 or opt.use_dense or opt.use_convnext:
            opt.linear_num = 1024
        elif opt.use_efficient:
            opt.linear_num = 1792
        elif opt.use_NAS:
            opt.linear_num = 4032
        else:
            opt.linear_num = 2048

    for iter, data in enumerate(dataloaders):
        img, label = data
        n, c, h, w = img.size()
        # count += n
        # print(count)
        pbar.update(n)
        ff = torch.FloatTensor(n,opt.linear_num).zero_().cuda()
        
        ori_img = img
        # print("ori_img.shape", ori_img.shape)

        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            for scale in ms:
                if scale != 1:
                    # bicubic is only  available in pytorch>= 1.1
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
                # input_img.shape = [32, 3, 256, 128]
                outputs = model(input_img) 
                # if count == 0:
                #     count = 1
                #     print(outputs.shape)
                ff += outputs
        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        
        if iter == 0:
            features = torch.FloatTensor(len(dataloaders.dataset), ff.shape[1])
            imgs = torch.FloatTensor(len(dataloaders.dataset), ori_img.shape[1], ori_img.shape[2], ori_img.shape[3]).zero_().cuda()
            # print(imgs.shape)
        #features = torch.cat((features,ff.data.cpu()), 0)
        start = iter*opt.batchsize
        end = min( (iter+1)*opt.batchsize, len(dataloaders.dataset))
        features[ start:end, :] = ff
        # print(ori_img.shape)
        imgs[start:end, :, :, :] = ori_img
    pbar.close()
    return features, imgs


def extract_single_feature(model, img, linear_num):
    ff = torch.FloatTensor(1, linear_num).zero_()
    ff = ff.to('cuda')
    img = img.to('cuda')
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


def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        #filename = path.split('/')[-1]
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels

def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc

def evaluate(qf,ql,qc,gf,gl,gc):
    # print("qc:", qc)
    query = qf.view(-1,1)
    # print(query.shape)
    score = torch.mm(gf,query)
    score = score.squeeze(1).cpu()
    score = score.detach().numpy()
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    # index = index[0:2000]
    # good index
    # qll = np.array(ql.item()).astype('int64')
    qll = np.array(ql).astype('int64')
    query_index = np.argwhere(gl==qll)
    qcc = np.array(qc).astype('int64')
    camera_index = np.argwhere(gc==qcc)

    # good_index: remain id with the same lable but different camera graphs in gallery
    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    # junk_index1: index which lable == -1
    junk_index1 = np.argwhere(gl==-1)
    # junk_index2: remain id with the same lable and same camera graphs in gallery
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1) #.flatten())
    
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp

gallery_path = image_datasets['gallery'].imgs
query_path = image_datasets['query'].imgs

gallery_cam, gallery_label = get_id(gallery_path)
query_cam, query_label = get_id(query_path)

model_structure = ft_net(opt.nclasses, stride=opt.stride, ibn=opt.ibn, linear_num=opt.linear_num)
model = load_network(model_structure)
# Remove the final fc layer and classifier layer
model.classifier.classifier = nn.Sequential()

# Change to test mode
model = model.eval()
if use_gpu:
    model = model.cuda()

# fuse conv and bn for faster inference
model = fuse_all_conv_bn(model)

with torch.no_grad():
    gallery_feature, gallery_imgs = extract_feature(model,dataloaders['gallery'])
    query_feature, query_imgs = extract_feature(model,dataloaders['query'])

if opt.use_FGSM:
    attacker = FGSM()

# attack_feature = attacker.get_attack_feature(
#     model=model, test_loader=test_loaders, epsilon=0.1, linear_num=opt.linear_num, 
#     gallery_feature=gallery_feature, gallery_label=gallery_label
# )
# print(attack_feature.shape)

query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()
gallery_imgs = gallery_imgs.cuda()
query_imgs = query_imgs.cuda()
print("query_imgs.shape", query_imgs.shape)

CMC = torch.IntTensor(len(gallery_label)).zero_()
ap = 0.0
print("get_attack_img and eval!")
pbar2 = tqdm()
count = 0

for i in range(len(query_label)):
    cur_img = query_imgs[i]
    cur_img = cur_img.unsqueeze(0)
    # attack mode
    attacked_img = attacker.get_attack_img(
        model, cur_img, query_label[i], opt.epsilon, opt.linear_num, gallery_feature, gallery_label
    )
    # original mode
    # attacked_img = cur_img
    attacked_feature = extract_single_feature(model, attacked_img, opt.linear_num)
    attacked_feature.detach()
    ap_tmp, CMC_tmp = evaluate(
        attacked_feature.squeeze(0), query_label[i], query_cam[i], gallery_feature, gallery_label, gallery_cam
    )
    count = count + 1
    if CMC_tmp[0] == -1:
        continue
    CMC = CMC + CMC_tmp
    ap += ap_tmp
pbar2.close()

print(CMC, CMC.sum())

CMC = CMC.float()
CMC = CMC / len(query_label)
print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9],ap / len(query_label)))