import cv2
import numpy as np
import torch
import scipy.io
import os
import pickle
from model import ft_net
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms
from fastreid_tools import get_model

# target id: 311

class Reid_Block:
    def __init__(self, model, gallery_feature, gallery_id, query_feature, query_id, 
                 gallery_cam, query_cam, adv_id, adv_img, adv_name, imp_id, trans=None):
        self.model = model
        self.gallery_feature = gallery_feature                        # torch.Size([19732, 751])
        self.gallery_id = gallery_id                                  # (19732, )
        self.query_feature = query_feature                            # torch.Size([3368, 751])
        self.query_id = query_id                                      # (3368, )
        self.gallery_cam = gallery_cam                                # (19732, )
        self.query_cam = query_cam                                    # (3368, )
        self.adv_id = adv_id                                          # String
        self.adv_img = adv_img
        self.adv_name = adv_name
        self.imp_id = imp_id
        self.trans = trans
        self.gallery_advmark = torch.tensor([1. if i == int(adv_id) else 0. for i in gallery_id], device="cuda")
        self.query_advmark = torch.tensor([1. if i == int(adv_id) else 0. for i in query_id], device="cuda")
        self.gallery_impmark = torch.tensor([1. if i == int(imp_id) else 0 for i in gallery_id], device="cuda")
        self.data_transforms = transforms.Compose([
            transforms.Resize((256, 128), interpolation=3),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def adding_pattern(self, adv_img, ground_img, trans=[-1, -1, -1, -1, -1, -1, -1, -1]):
        """
        This function is used to add the pattern (which generated by diffusion model) to the reid image.

        Args:
            adv_img (PIL.Image): the image which is used to paste on person's clothes
            ground_img (PIL.Image): the reid image in Market
        Return:
            transformed_image_pil (PIL.Image): the final reid image with pattern on person's clothes
        """
        # trans = [0, 15, 0, 35, 50, 50, 50, 0] # only for testing
        if trans[0] == -1:
            return ground_img
        adv_img_cv2 = cv2.cvtColor(np.asarray(adv_img), cv2.COLOR_RGB2BGR)
        ground_img_cv2 = cv2.cvtColor(np.asarray(ground_img), cv2.COLOR_RGB2BGR)
        adv_img_cv2 = cv2.resize(adv_img_cv2, (64, 64))
        transparent_image = np.zeros((128, 64, 4), dtype=np.uint8)
        transparent_image[:64, :64, :3] = adv_img_cv2
        src_list = np.float32([(0, 0), (0, 63), (63, 63), (63, 0)])
        end_list = np.float32([(trans[0], trans[1]), (trans[2], trans[3]), (trans[4], trans[5]), (trans[6], trans[7])])
        matrix = cv2.getPerspectiveTransform(src_list, end_list)
        transformed_image = cv2.warpPerspective(transparent_image, matrix, (64, 128))
        mask = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]
        transformed_image[:, :, 3] = mask
        transformed_image_pil = Image.fromarray(cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB))
        transformed_image_pil
        for i in range(transformed_image.shape[0]):
            for j in range(transformed_image.shape[1]):
                if transformed_image[i, j, 3] > 0:
                    ground_img_cv2[i, j, :3] = transformed_image[i, j, :3]
        transformed_image_pil = Image.fromarray(cv2.cvtColor(ground_img_cv2, cv2.COLOR_BGR2RGB))
        return transformed_image_pil

    
    def adding_tensor_pattern(self, adv_tensor, ground_tensor, trans=[-1, -1, -1, -1, -1, -1, -1, -1]):
        if trans[0] == -1:
            return ground_tensor
        assert ground_tensor.shape == torch.Size([3, 128, 64]), "the shape of reid image tensor is not proper"
        transparent_image = np.zeros((128, 64, 4), dtype=np.uint8)
        color_image = np.full((64, 64, 3), 10)
        # get mask
        transparent_image[:64, :64, :3] = color_image
        src_list = np.float32([(0, 0), (0, 63), (63, 63), (63, 0)])
        end_list = np.float32([(trans[0], trans[1]), (trans[2], trans[3]), (trans[4], trans[5]), (trans[6], trans[7])])
        matrix = cv2.getPerspectiveTransform(src_list, end_list)
        transformed_image = cv2.warpPerspective(transparent_image, matrix, (64, 128))
        mask = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]
        mask_tensor = torch.tensor(mask)
        mask_tensor = mask_tensor / 255
        mask_tensor.unsqueeze(dim=2)
        mask_tensor = mask_tensor.repeat(3, 1, 1).unsqueeze(0).cuda()
        # perspective process
        src_list2 = np.float32([(0, 0), (0, 127), (63, 127), (63, 0)])
        adv_tensor = adv_tensor.unsqueeze(0)
        adv_tensor = F.interpolate(adv_tensor, (128, 64), mode='bilinear')
        adv_tensor.squeeze(0)
        new_tensor = transforms.functional.perspective(adv_tensor, src_list2, end_list)
        result_tensor = new_tensor * mask_tensor + ground_tensor * (1 - mask_tensor)
        result_tensor = result_tensor.squeeze(0)
        return result_tensor
        
    def get_adv_feature(self, adv_image):
        assert len(self.adv_name) == len(self.trans), "the length of adv_name and trans should be the same"
        adv_list = []
        for i in range(len(self.adv_img)):
            adv_list.append(self.adding_pattern(adv_image, self.adv_img[i], self.trans[self.adv_name[i]]))
            # adv_list[i].save("After:" + adv_name[i])
        adv_tensor = [self.data_transforms(x).cuda() for x in adv_list]
        adv_tensor = torch.stack(adv_tensor)                        # torch.Size(n, 3, 256, 128)
        adv_feature = self.model(adv_tensor) # torch.Size(n, 751)
        return adv_feature
    
    def get_adv_feature_from_tensor(self, adv_tensor):
        assert len(self.adv_name) == len(self.trans), "the length of adv_name and trans should be the same"
        tensor_list = []
        reid_trans = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Resize((128, 64))
        ])
        norm_trans = transforms.Compose([
            transforms.Resize((256, 128), interpolation=3),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        for i in range(len(self.adv_img)):
            added_tensor = self.adding_tensor_pattern(adv_tensor, reid_trans(self.adv_img[i]).cuda(), self.trans[self.adv_name[i]])
            # only for testing
            # adv_Image = transforms.ToPILImage()(added_tensor.squeeze(0)).convert('RGB')
            # adv_Image.save("TensorTrans_" + self.adv_name[i])
            tensor_list.append(norm_trans(added_tensor))
        tensor_list = torch.stack(tensor_list)
        tensor_list = tensor_list.cuda()
        adv_feature = self.model(tensor_list)
        return adv_feature
        
    def evading_identity_loss(self, adv_feature):
        adv_id = torch.nonzero(self.gallery_advmark).squeeze()
        adv_gallery_feature = self.gallery_feature[adv_id]
        score = F.cosine_similarity(adv_feature.unsqueeze(1), adv_gallery_feature.unsqueeze(0), dim=2) # torch.Size(n, adv)
        similarity_loss = torch.sum(score)
        return similarity_loss
    
    def evading_pair_generator(self, batch_size):
        adv_id = torch.nonzero(self.query_advmark).squeeze()
        left_id = []
        right_id = []
        while len(left_id) < batch_size:
            indices = torch.randperm(adv_id.size(0))[:2]
            if indices[0] == indices[1]:
                continue
            selected_id = adv_id[indices]
            if self.query_cam[selected_id[0]] == self.query_cam[selected_id[1]]:
                continue
            left_id.append(selected_id[0].item() - adv_id[0].item())
            right_id.append(selected_id[1].item() - adv_id[0].item())
        return left_id, right_id
    
    def evading_camera_loss(self, adv_feature):
        """
            hyperparameters:
        """
        batch_size = 32
        left_id, right_id = self.evading_pair_generator(batch_size)
        score_list = []
        for i in range(len(left_id)):
            score_list.append(F.cosine_similarity(adv_feature[left_id[i]].unsqueeze(0).unsqueeze(1), adv_feature[right_id[i]].unsqueeze(0).unsqueeze(0), dim=2))
        score_list = torch.stack(score_list)
        return torch.sum(score_list)
    
    def evading_loss_image(self, adv_image):
        assert isinstance(adv_image, Image.Image), "the type of adv_image should be PIL.Image!"
        adv_feature = self.get_adv_feature(adv_image)
        return self.evading_identity_loss(adv_feature), self.evading_camera_loss(adv_feature)
    
    def evading_maxmin_loss(self, adv_feature):
        adv_id = torch.nonzero(self.gallery_advmark).squeeze()
        score = F.cosine_similarity(adv_feature.unsqueeze(1), self.gallery_feature.unsqueeze(0), dim=2) # torch.Size([5, 19732])
        adv_max, _ = torch.max(score[:, adv_id], dim=1)
        non_adv_id = torch.cat([torch.arange(adv_id[0].item()), torch.arange(adv_id[-1].item() + 1, score.size(1))])
        not_adv_min, _ = torch.min(score[:, non_adv_id], dim=1)
        diff = torch.relu(adv_max - not_adv_min) * 100
        loss = torch.mean(diff ** 2)
        return loss
    
    def evading_loss(self, adv_tensor):
        adv_feature = self.get_adv_feature_from_tensor(adv_tensor)
        return self.evading_camera_loss(adv_feature)
    
    def impersonating_identity_loss(self, adv_tensor):
        pass
    
    def impersonating_pair_generator(self, batch_size):
        adv_id = torch.nonzero(self.query_advmark).squeeze()
        imp_id = torch.nonzero(self.gallery_impmark).squeeze()
        left_id = []
        right_id = []
        target_id = []
        while len(left_id) < batch_size:
            indices = torch.randperm(adv_id.size(0))[:2]
            if indices[0] == indices[1]:
                continue
            selected_adv = adv_id[indices]
            if self.query_cam[selected_adv[0]] == self.query_cam[selected_adv[1]]:
                continue
            targets = torch.randperm(imp_id.size(0))[:1]
            selected_imp = imp_id[targets]
            left_id.append(selected_adv[0].item() - adv_id[0].item())
            right_id.append(selected_adv[1].item() - adv_id[0].item())
            target_id.append(selected_imp[0].item())
        return left_id, right_id, target_id
    
    def impersonating_camera_loss(self, adv_feature):
        # hyper
        batch_size = 32
        left_id, right_id, target_id = self.impersonating_pair_generator(batch_size)
        score_list = []
        for i in range(len(left_id)):
            score01 = F.cosine_similarity(adv_feature[left_id[i]].unsqueeze(0).unsqueeze(1), self.gallery_feature[target_id[i]].unsqueeze(0).unsqueeze(0), dim=2)
            score02 = F.cosine_similarity(adv_feature[right_id[i]].unsqueeze(0).unsqueeze(1), self.gallery_feature[target_id[i]].unsqueeze(0).unsqueeze(0), dim=2)
            score12 = F.cosine_similarity(adv_feature[left_id[i]].unsqueeze(0).unsqueeze(1), adv_feature[right_id[i]].unsqueeze(0).unsqueeze(0), dim=2)
            score_list.append(score12 - score01 - score02)
        return torch.stack(score_list).sum()
    
    def impersonating_maxmin_loss(self, adv_feature):
        imp_id = torch.nonzero(self.gallery_impmark).squeeze()
        adv_id = torch.nonzero(self.gallery_advmark).squeeze()
        score = F.cosine_similarity(adv_feature.unsqueeze(1), self.gallery_feature.unsqueeze(0), dim=2) # torch.Size([5, 19732])
        imp_max, _ = torch.max(score[:, imp_id], dim=1)
        non_imp_id = torch.cat([torch.arange(imp_id[0].item()), torch.arange(imp_id[-1].item() + 1, score.size(1))])
        not_imp_max, _ = torch.max(score[:, non_imp_id], dim=1)
        diff = torch.relu(not_imp_max - imp_max) * 10000
        loss = torch.mean(diff ** 2)
        return loss
    
    def impersonating_loss(self, adv_tensor):
        adv_feature = self.get_adv_feature_from_tensor(adv_tensor)
        return self.impersonating_camera_loss(adv_feature)
        
    def evaluate(self, adv_image=None):
        tensor_list = []
        reid_trans = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Resize((128, 64))
        ])
        norm_trans = transforms.Compose([
            transforms.Resize((256, 128), interpolation=3),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        if adv_image is not None:
            adv_tensor = transforms.ToTensor()(adv_image)
            for i in range(len(self.adv_img)):
                added_tensor = self.adding_tensor_pattern(adv_tensor.cuda(), reid_trans(self.adv_img[i]).cuda(), self.trans[self.adv_name[i]])
                tensor_list.append(norm_trans(added_tensor))
            tensor_list = torch.stack(tensor_list)
            tensor_list = tensor_list.cuda()
        else:
            for i in range(len(self.adv_img)):
                tensor_list.append(norm_trans(reid_trans(self.adv_img[i])))
            tensor_list = torch.stack(tensor_list)
            tensor_list = tensor_list.cuda()
        
        adv_feature = self.model(tensor_list)
        id_list = []
        rk_list = []
        # sim = F.cosine_similarity(adv_feature.unsqueeze(1), self.gallery_feature.unsqueeze(0), dim=2)
        for i in range(len(self.adv_img)):
            sim = F.cosine_similarity(adv_feature[i].unsqueeze(0).unsqueeze(1), self.gallery_feature.unsqueeze(0), dim=2)
            _, idx = sim.topk(10, dim=1)
            idx = idx.to("cpu")
            img_id_list = [self.gallery_id[x] for x in idx]
            id_list.append(img_id_list)
            _, idx2 = sim.topk(19732, dim=1)
            idx2 = idx2.to("cpu")
            img_id_list2 = [self.gallery_id[x] for x in idx2]
            target_rk = []
            for j in range(len(img_id_list2[0])):
                if img_id_list2[0][j] == int(self.imp_id):
                    target_rk.append(j + 1)
            rk_list.append(target_rk)
            
        # _, top_idx = sim.topk(10, dim=1)        
        # for i in range(len(self.adv_img)):
        #     img_id_list = [self.gallery_id[x] for x in top_idx[i]]
        #     id_list.append(img_id_list)
        print(id_list)
        print(rk_list)
            
    
if __name__ == "__main__":
    # x = np.full((3, 2), 4)
    # y = torch.tensor(x)
    # print(y)
    # raise NotImplementedError
    result = scipy.io.loadmat('./inter_veriable/gallery_info_240324.mat')
    query_feature = torch.FloatTensor(result['query_f']).cuda()              # torch.Size([3368, 751])
    query_cam = result['query_cam'][0]                                       # (3368, )
    query_label = result['query_label'][0]                                   # (3368, )
    gallery_feature = torch.FloatTensor(result['gallery_f']).cuda()          # torch.Size([19732, 751])
    gallery_cam = result['gallery_cam'][0]                                   # (19732, )
    gallery_label = result['gallery_label'][0]                               # (19732, )
    
    # model = ft_net()
    # model_path = './inter_veriable/net_last.pth'
    # checkpoint = torch.load(model_path)
    # model.load_state_dict(checkpoint)
    # model.to("cuda")
    
    # model.eval()
    # del checkpoint
    model = get_model()
    
    target_img = []
    adv_name = []
    adversarial = "0311"
    id_path = "/mnt/Data_DSG_2/xueyuhao/Market/pytorch/query/" + adversarial
    id_imgs = os.listdir(id_path)
    trans_path = "inter_veriable/market_trans/" + adversarial + ".mat"
    with open(trans_path, 'rb') as f:
        trans_data = pickle.load(f)
    for img in id_imgs:
        img_path = id_path + '/' + img
        target_img.append(Image.open(img_path))
        adv_name.append(img)
    
    test_reid = Reid_Block(model, gallery_feature, gallery_label, query_feature, query_label, 
                           gallery_cam, query_cam, adversarial, target_img, adv_name, "0502", trans_data)
    
    test_adv_image = Image.open("outputs/txt2img-samples/grid-0043.png")
    test_trans = transforms.Compose([
        transforms.Resize((256, 128), interpolation=3),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_adv_tensor = test_trans(test_adv_image)
    # test_adv_tensor.requires_grad_()
    x = test_reid.impersonating_loss(test_adv_tensor.cuda())
    print("x:", x)
    test_reid.evaluate()
    print("-----")
    test_reid.evaluate(test_adv_image)
    