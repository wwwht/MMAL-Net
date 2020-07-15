import sys
sys.path.append('..')
import torch
from torch import nn
import torch.nn.functional as F
from networks import resnet
from config import pretrain_path, coordinates_cat, iou_threshs, window_nums_sum, ratios, N_list, max_checkpoint_num, proposalN
# from config import max_checkpoint_num, proposalN, eval_trainset, set
import numpy as np
from utils.AOLM import AOLM
import cv2
from PIL import Image
from torchvision import transforms
import ipdb
import os

def save_image(tensor):
    dir = 'results'
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    image.save('results.jpg')
def nms(scores_np, proposalN, iou_threshs, coordinates):
    if not (type(scores_np).__module__ == 'numpy' and len(scores_np.shape) == 2 and scores_np.shape[1] == 1):
        raise TypeError('score_np is not right')

    windows_num = scores_np.shape[0]
    indices_coordinates = np.concatenate((scores_np, coordinates), 1)

    indices = np.argsort(indices_coordinates[:, 0])
    indices_coordinates = np.concatenate((indices_coordinates, np.arange(0,windows_num).reshape(windows_num,1)), 1)[indices]                  #[339,6]
    indices_results = []

    res = indices_coordinates

    while res.any():
        indice_coordinates = res[-1]
        indices_results.append(indice_coordinates[5])

        if len(indices_results) == proposalN:
            return np.array(indices_results).reshape(1,proposalN).astype(np.int)
        res = res[:-1]

        # Exclude anchor boxes with selected anchor box whose iou is greater than the threshold
        start_max = np.maximum(res[:, 1:3], indice_coordinates[1:3])
        end_min = np.minimum(res[:, 3:5], indice_coordinates[3:5])
        lengths = end_min - start_max + 1
        intersec_map = lengths[:, 0] * lengths[:, 1]
        intersec_map[np.logical_or(lengths[:, 0] < 0, lengths[:, 1] < 0)] = 0
        iou_map_cur = intersec_map / ((res[:, 3] - res[:, 1] + 1) * (res[:, 4] - res[:, 2] + 1) +
                                      (indice_coordinates[3] - indice_coordinates[1] + 1) *
                                      (indice_coordinates[4] - indice_coordinates[2] + 1) - intersec_map)
        res = res[iou_map_cur <= iou_threshs]

    while len(indices_results) != proposalN:
        indices_results.append(indice_coordinates[5])

    return np.array(indices_results).reshape(1, -1).astype(np.int)

class APPM(nn.Module):
    def __init__(self):
        super(APPM, self).__init__()
        self.avgpools = [nn.AvgPool2d(ratios[i], 1) for i in range(len(ratios))]

    def forward(self, proposalN, x, ratios, window_nums_sum, N_list, iou_threshs, DEVICE='cuda'):
        batch, channels, _, _ = x.size()
        avgs = [self.avgpools[i](x) for i in range(len(ratios))]

        # feature map sum
        fm_sum = [torch.sum(avgs[i], dim=1) for i in range(len(ratios))]

        all_scores = torch.cat([fm_sum[i].view(batch, -1, 1) for i in range(len(ratios))], dim=1)
        windows_scores_np = all_scores.data.cpu().numpy()
        window_scores = torch.from_numpy(windows_scores_np).to(DEVICE).reshape(batch, -1)

        # nms
        proposalN_indices = []
        for i, scores in enumerate(windows_scores_np):
            indices_results = []
            for j in range(len(window_nums_sum)-1):
                indices_results.append(nms(scores[sum(window_nums_sum[:j+1]):sum(window_nums_sum[:j+2])], proposalN=N_list[j], iou_threshs=iou_threshs[j],
                                           coordinates=coordinates_cat[sum(window_nums_sum[:j+1]):sum(window_nums_sum[:j+2])]) + sum(window_nums_sum[:j+1]))
            # indices_results.reverse()
            proposalN_indices.append(np.concatenate(indices_results, 1))   # reverse

        proposalN_indices = np.array(proposalN_indices).reshape(batch, proposalN)
        proposalN_indices = torch.from_numpy(proposalN_indices).to(DEVICE)
        proposalN_windows_scores = torch.cat(
            [torch.index_select(all_score, dim=0, index=proposalN_indices[i]) for i, all_score in enumerate(all_scores)], 0).reshape(
            batch, proposalN)

        return proposalN_indices, proposalN_windows_scores, window_scores

class MainNet(nn.Module):
    def __init__(self, proposalN, num_classes, channels):
        '''
        主要网络结构，proposalN box的数量
        '''
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        super(MainNet, self).__init__()
        self.num_classes = num_classes
        self.proposalN = proposalN
        self.pretrained_model = resnet.resnet50(pretrained=True, pth_path='/home/junyiwu1688/.cache/torch/checkpoints/resnet50-19c8e357.pth')
        self.rawcls_net = nn.Linear(channels, num_classes)
        self.APPM = APPM()

    def forward(self, x, epoch, batch_idx, status='test', DEVICE='cuda'):
        # fm : 
        fm, embedding, conv5_b = self.pretrained_model(x)
        batch_size, channel_size, side_size, _ = fm.shape
        assert channel_size == 2048

        # raw branch
        # 粗粒度的分类
        raw_logits = self.rawcls_net(embedding)

        #SCDA
        coordinates = torch.tensor(AOLM(fm.detach(), conv5_b.detach()))
        # ipdb.set_trace()

        local_imgs = torch.zeros([batch_size, 3, 448, 448]).to(DEVICE)  # [N, 3, 448, 448]
        for i in range(batch_size):
            [x0, y0, x1, y1] = coordinates[i]
            local_imgs[i:i + 1] = F.interpolate(x[i:i + 1, :, x0:(x1+1), y0:(y1+1)], size=(448, 448),
                                                mode='bilinear', align_corners=True)  # [N, 3, 224, 224]
        save_image(local_imgs)
        local_fm, local_embeddings, _ = self.pretrained_model(local_imgs.detach())  # [N, 2048]
        local_logits = self.rawcls_net(local_embeddings)  # [N, 200]

        proposalN_indices, proposalN_windows_scores, window_scores \
            = self.APPM(self.proposalN, local_fm.detach(), ratios, window_nums_sum, N_list, iou_threshs, DEVICE)

        if status == "train":
            # window_imgs cls
            window_imgs = torch.zeros([batch_size, self.proposalN, 3, 224, 224]).to(DEVICE)  # [N, 4, 3, 224, 224]
            for i in range(batch_size):
                for j in range(self.proposalN):
                    [x0, y0, x1, y1] = coordinates_cat[proposalN_indices[i, j]]
                    window_imgs[i:i + 1, j] = F.interpolate(local_imgs[i:i + 1, :, x0:(x1 + 1), y0:(y1 + 1)], size=(224, 224),
                                                                mode='bilinear',
                                                                align_corners=True)  # [N, 4, 3, 224, 224]

            window_imgs = window_imgs.reshape(batch_size * self.proposalN, 3, 224, 224)  # [N*4, 3, 224, 224]
            _, window_embeddings, _ = self.pretrained_model(window_imgs.detach())  # [N*4, 2048]
            proposalN_windows_logits = self.rawcls_net(window_embeddings)  # [N* 4, 200]
        else:
            proposalN_windows_logits = torch.zeros([batch_size * self.proposalN, self.num_classes]).to(DEVICE)

        return proposalN_windows_scores, proposalN_windows_logits, proposalN_indices, \
               window_scores, coordinates, raw_logits, local_logits, local_imgs


if __name__ == '__main__':

    # images = torch.rand(1,3,448,448).cuda()
    img = Image.open('./test1.jpg').convert('RGB')
    # img = Image.fromarray(img, mode='RGB')
    img = transforms.Resize((448, 448), Image.BILINEAR)(img)
    # img = transforms.CenterCrop(self.input_size)(img)
    img = transforms.ToTensor()(img)
    img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
    img = img.unsqueeze(0).cuda()


    Model =  MainNet(proposalN=proposalN, num_classes=10, channels=2048).cuda()
    
    proposalN_windows_score, proposalN_windows_logits, indices, \
            window_scores, _, raw_logits, local_logits, _ = Model(img, 1, 1, 'train')

    print('proposalN_windows_score.shape**********')
    print(proposalN_windows_score.shape)
    print('proposalN_windows_logits***********')
    print(proposalN_windows_logits)
    print('indices**********')
    print(indices)
    print('window_scores.shape***********')
    print(window_scores.shape)
    print('raw_logits**********')
    print(raw_logits)
    print('local_logits**********')
    print(local_logits)