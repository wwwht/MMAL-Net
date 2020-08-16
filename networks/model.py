import sys
sys.path.append('..')
import torch
from torch import nn
import torch.nn.functional as F
from networks import resnet
from config import pretrain_path, coordinates_cat, iou_threshs, window_nums_sum, ratios, N_list, max_checkpoint_num, proposalN, \
window_nums_sum2, ratios2, N_list2, proposalN2,coordinates_cat2
# from config import max_checkpoint_num, proposalN, eval_trainset, set
import numpy as np
from utils.AOLM import AOLM
import cv2
from PIL import Image
from torchvision import transforms
import ipdb
import os
from utils.vis import *

# class mix_feature(nn.Module):
#     def __init__(self,in_cannel, out_channel):
#         self.con1x1 = self.conv_batch()

#     def conv_batch(in_num, out_num, kernel_size=3, padding=1, stride=1):
#         return nn.Sequential(
#             nn.Conv2d(in_num, out_num, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
#             nn.BatchNorm2d(out_num),
#             nn.LeakyReLU())

def mix_features(low_level, high_level):
    '''
    low_level features有更高的分辨率和细节信息 [128,28,28]
    high_level features 有较低的分辨了和语义信息[256,14,14]
    '''
    low_channel = low_level.shape[1]
    low_size = low_level.shape[2]
    high_channel = high_level.shape[1]
    high_size = high_level.shape[2]

    high_level_resize = F.interpolate(high_level, scale_factor = low_size/high_size, mode = "bilinear")

    mix_feature = torch.cat([low_level, high_level_resize], dim=1)
    return mix_feature

def save_image(tensor, name):
    dir = 'results'
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    image.save('results_{}.jpg'.format(name))
def nms(scores_np, proposalN, iou_threshs, coordinates):
    '''
    scores_np: (241, 1)
    proposal_N: int, [2, 3, 2]其中之一
    iou_threshs: int ,0.25
    coordinates: 对应ratio的滑动窗口们的坐标（241个滑动窗口）
    '''
    if not (type(scores_np).__module__ == 'numpy' and len(scores_np.shape) == 2 and scores_np.shape[1] == 1):
        raise TypeError('score_np is not right')

    windows_num = scores_np.shape[0] # 窗口数目
    # 将分数和坐标
    indices_coordinates = np.concatenate((scores_np, coordinates), 1) # coordinates(241, 4)， scores_np(241,1)--> (241,5)

    indices = np.argsort(indices_coordinates[:, 0]) # 从小到大排序的索引
    indices_coordinates = np.concatenate((indices_coordinates, np.arange(0,windows_num).reshape(windows_num,1)), 1)[indices] 
    # 将分数-坐标-index拼接起来
    indices_results = []

    res = indices_coordinates

    while res.any():
        indice_coordinates = res[-1] # 从最后一个开始，因为最后一个
        indices_results.append(indice_coordinates[5])

        if len(indices_results) == proposalN:
            return np.array(indices_results).reshape(1,proposalN).astype(np.int)
        res = res[:-1] # 取出除最后一个之前的

        # Exclude anchor boxes with selected anchor box whose iou is greater than the threshold
        # 去掉和indice_coordinates iou大于阈值的box
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
    def __init__(self,ratio, concats):
        super(APPM, self).__init__()
        self.concats = concats
        self.ratio = ratio
        self.avgpools = [nn.AvgPool2d(self.ratio[i], 1) for i in range(len(self.ratio))] # 不同的池化卷积层，在特征图上
        # ipdb.set_trace()
    def forward(self, proposalN, x, ratios, window_nums_sum, N_list, iou_threshs, DEVICE='cuda'):
        # window_nums_sum： 滑动窗口的数量[0, 241, 235, 115]，、
        # ratios: 特征图上滑动窗口的大小
        batch, channels, _, _ = x.size()
        avgs = [self.avgpools[i](x) for i in range(len(self.ratio))]

        # feature map sum
        fm_sum = [torch.sum(avgs[i], dim=1) for i in range(len(self.ratio))] # 将2048个通道相加，得到使用不同滑动窗口得到的activation map 

        all_scores = torch.cat([fm_sum[i].view(batch, -1, 1) for i in range(len(self.ratio))], dim=1) # 不同滑动窗口的激活图concat
        windows_scores_np = all_scores.data.cpu().numpy() # (10, 591, 1)
        window_scores = torch.from_numpy(windows_scores_np).to(DEVICE).reshape(batch, -1) # torch.Size([10, 591])

        # nms
        proposalN_indices = [] # 
        for i, scores in enumerate(windows_scores_np): # scores [591,1]
            indices_results = []
            for j in range(len(window_nums_sum)-1):
                # 最终我们期望留下的bbox类型有3种，将每一种bbox的滑动窗口进行nms，
                indices_results.append(nms(scores[sum(window_nums_sum[:j+1]):sum(window_nums_sum[:j+2])], proposalN=N_list[j], iou_threshs=iou_threshs[j],
                coordinates=self.concats[sum(window_nums_sum[:j+1]):sum(window_nums_sum[:j+2])]) + sum(window_nums_sum[:j+1]))
            # indices_results.reverse()
            proposalN_indices.append(np.concatenate(indices_results, 1))   # reverse
        
        proposalN_indices = np.array(proposalN_indices).reshape(batch, proposalN)
        proposalN_indices = torch.from_numpy(proposalN_indices).to(DEVICE)
        proposalN_windows_scores = torch.cat(
            [torch.index_select(all_score, dim=0, index=proposalN_indices[i]) for i, all_score in enumerate(all_scores)], 0).reshape(
            batch, proposalN)
        # ipdb.set_trace()
        return proposalN_indices, proposalN_windows_scores, window_scores

class MainNet(nn.Module):
    def __init__(self, proposalN, num_classes, channels):
        '''
        主要网络结构，proposalN box的数量
        '''
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        super(MainNet, self).__init__()
        self.num_classes = num_classes # 类别
        self.proposalN = proposalN     # 预先定义的bbox 数量
        self.pretrained_model = resnet.resnet50(pretrained=True, pth_path='/home/junyiwu1688/.cache/torch/checkpoints/resnet50-19c8e357.pth')
        self.rawcls_net = nn.Linear(channels, num_classes)
        self.APPM = APPM(ratio = ratios,concats = coordinates_cat)
        self.APPM2 = APPM(ratio=ratios2, concats = coordinates_cat2)

    def forward(self, x, epoch, batch_idx, status='test', DEVICE='cuda'):
        # fm : 
        fm, embedding, conv5_b = self.pretrained_model(x)
        batch_size, channel_size, side_size, _ = fm.shape
        assert channel_size == 2048
        # ipdb.set_trace()
        # mix_feature = 
        # raw branch
        # 粗粒度的分类
        raw_logits = self.rawcls_net(embedding)
        ############################################################################################################
        #SCDA 找到局部图像
        # coordinates = torch.tensor(AOLM(fm.detach(), conv5_b.detach()))
        # image = image_with_boxes(x, coordinates=coordinates)
        fm = mix_features(conv5_b, fm)
        # ipdb.set_trace()
        proposalN_indices2, proposalN_windows_scores2, window_scores2 \
            = self.APPM2(proposalN2, fm.detach(), ratios2, window_nums_sum2, N_list2, iou_threshs, DEVICE)

        window_imgs = torch.zeros([batch_size, proposalN2, 3, 448, 448]).to(DEVICE)  # [N, 4, 3, 224, 224]
        for i in range(batch_size):
            for j in range(proposalN2):
                [x0, y0, x1, y1] = coordinates_cat2[proposalN_indices2[i, j]]
                # ipdb.set_trace()
                window_imgs[i:i+1, j] = x[i:i + 1, :, x0:(x1), y0:(y1)]
        window_imgs = window_imgs.reshape(batch_size * proposalN2, 3, 448, 448)  # [N*3, 3, 224, 224]
        ############################################找到注意力区域####################################################
        # for i,img in enumerate(window_imgs):
        #     save_image(img, str(i))
        local_fm, local_embeddings, _ = self.pretrained_model(window_imgs.detach())  # [N, 2048]
        local_logits = self.rawcls_net(local_embeddings)  # [N*3, 200]
        local_logits_split = local_logits.split(3,0)
        ipdb.set_trace()
        # 

        # APPM通过局部图像和滑动框,选择特征图
        # proposalN_indices: 最终的bbox的 index
        # proposalN_windows_scores：bbox的分数
        # window_scores：所有bbox的分数



        return proposalN_windows_scores, proposalN_windows_logits, proposalN_indices, \
               window_scores, coordinates, raw_logits, local_logits, local_imgs


if __name__ == '__main__':

    img = torch.rand(3,3,768,768).cuda()
    # img = Image.open('./test4.jpg').convert('RGB')
    # img = Image.fromarray(img, mode='RGB')
    # img = transforms.RandomCrop((768, 768), Image.BILINEAR)(img)
    # img = transforms.CenterCrop(self.input_size)(img)
    # img = transforms.ToTensor()(img)
    # img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
    # img = img.unsqueeze(0).cuda()


    Model =  MainNet(proposalN=proposalN, num_classes=1, channels=2048).cuda()
    
    proposalN_windows_score, proposalN_windows_logits, indices, \
            window_scores, coordinates, raw_logits, local_logits, _ = Model(img, 1, 1, 'train')

    

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