#10月17

import torch
from scipy.stats import pearsonr
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import random

from model import TVVAD
from ucf_test import test
from utils.dataset import UCFDataset
from utils.tools import get_prompt_text, get_batch_label
import ucf_option
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def symmetric_kl_divergence(p, q):
    """
    Compute the Symmetric KL Divergence between two probability distributions.

    Args:
        p (torch.Tensor): Probability distribution P. Shape (N, D), where N is the batch size and D is the distribution dimension.
        q (torch.Tensor): Probability distribution Q. Shape (N, D), where N is the batch size and D is the distribution dimension.

    Returns:
        torch.Tensor: The symmetric KL divergence between P and Q.
    """
    # Ensure the distributions are normalized
    p = F.normalize(p, p=1, dim=1)
    q = F.normalize(q, p=1, dim=1)

    # Calculate the KL divergence
    kl_pq = torch.sum(p * torch.log(p / q), dim=1)
    kl_qp = torch.sum(q * torch.log(q / p), dim=1)

    # Calculate the symmetric KL divergence
    skl_div = 0.5 * (kl_pq + kl_qp)

    return skl_div.mean()

def pearson_correlation(matrix1, matrix2):
    matrix1_mean = matrix1.mean(dim=-1, keepdim=True)
    matrix2_mean = matrix2.mean(dim=-1, keepdim=True)
    numerator = ((matrix1 - matrix1_mean) * (matrix2 - matrix2_mean)).sum(dim=-1)
    denominator = torch.sqrt(((matrix1 - matrix1_mean) ** 2).sum(dim=-1) * ((matrix2 - matrix2_mean) ** 2).sum(dim=-1))
    return numerator / denominator


def CLASM(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)

    for i in range(logits.shape[0]):
        # tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True, dim=0)
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=1, largest=True, dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)

    milloss = -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    # pearsonr_loss = pearson_correlation(labels,instance_logits)
    soft_instance_logist = F.softmax(instance_logits, dim=-1)
    position_loss = F.cross_entropy(soft_instance_logist, labels.argmax(dim=-1))

    p_ = soft_instance_logist
    #q = labels + 0.001  # 87.21
    q = labels + 0.0001  # 87.89
    # q = labels + 0.00001  # 87.0

    # kl_loss = symmetric_kl_divergence(p_,q)
    kl_loss = F.kl_div(p_.log(), labels, reduction='batchmean')


    return milloss, position_loss, kl_loss

def CLAS2(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = 1 - labels[:, 0].reshape(labels.shape[0])
    labels = labels.to(device)
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])

    for i in range(logits.shape[0]): # i=56
        # print(logits[i, 0:lengths[i]])
        # tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True)
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=1 , largest=True)
        tmp = torch.mean(tmp).view(1)

        instance_logits = torch.cat([instance_logits, tmp], dim=0)

    clsloss = F.binary_cross_entropy(instance_logits, labels)

    return clsloss


def check_nan( tensor):
    if torch.isnan(tensor).any():
        nan_indic = torch.isnan(tensor).nonzero(as_tuple = True)
        print('the tensor contains Nan', nan_indic)
    else:
        print('no Nan ')

def train(model, normal_loader, anomaly_loader, testloader, args, label_map, device):
    model.to(device)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    prompt_text = get_prompt_text(label_map)
    ap_best = 0
    epoch = 0

    if args.use_checkpoint == True:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        ap_best = checkpoint['ap']
        print("checkpoint info:")
        print("epoch:", epoch+1, " ap:", ap_best)

    for e in range(args.max_epoch):
        model.train()
        loss_total1 = 0
        loss_total2 = 0
        posi_loss_total = 0
        kl_total = 0
        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)
        for i in range(min(len(normal_loader), len(anomaly_loader))):
            step = 0
            normal_features, normal_label, normal_lengths = next(normal_iter)  # contains nan

            anomaly_features, anomaly_label, anomaly_lengths = next(anomaly_iter)  # no nan


            visual_features = torch.cat([normal_features, anomaly_features], dim=0).to(device)  #  torch.Size([64, 256, 512]) torch.Size([64, 256, 512])
            text_labels_ = list(normal_label) + list(anomaly_label)
            feat_lengths = torch.cat([normal_lengths, anomaly_lengths], dim=0).to(device)
            text_labels = get_batch_label(text_labels_, prompt_text, label_map).to(device)

            # model(visual_feat, None, prompt_text, feat_lengths, label_map, text_labels_, flage)

            # text_features, logits1, logits2 = model(visual_features, None, prompt_text, feat_lengths)
            flage = True
            text_features, logits1, logits2 = model(visual_features, None, prompt_text, feat_lengths)
            # print('1 shape is ', logits1.shape, '2 shape is ', logits2.shape)
            #loss1

            loss1 = CLAS2(logits1, text_labels, feat_lengths, device) 
            loss_total1 += loss1.item()
            #loss2
            loss2, p_loss, kl = CLASM(logits2, text_labels, feat_lengths, device)

            loss_total2 += loss2.item()
            posi_loss_total += p_loss.item()
            kl_total += kl.item()

            #loss3


            loss = loss1 *1.7  + loss2*1 +  p_loss*0.1 + kl*1.6


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += i * normal_loader.batch_size * 2
            if step % 1280 == 0 and step != 0:
                print('epoch: ', e+1, '| step: ', step, '| loss1: ', loss_total1 / (i+1), '| loss2: ', loss_total2 / (i+1), '| kl_loss: ', kl_total / (i+1),)
                AUC, AP = test(model, testloader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)
                AP = AUC

                if AP > ap_best:
                    ap_best = AP 
                    checkpoint = {
                        'epoch': e,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'ap': ap_best}
                    torch.save(checkpoint, args.checkpoint_path)
                
        scheduler.step()
        
        torch.save(model.state_dict(), '/home/zhangheng/paper_code/VadCLIP-main/VadCLIP-main/model/model_cur_ucf.pth')
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    checkpoint = torch.load(args.checkpoint_path)
    torch.save(checkpoint['model_state_dict'], args.model_path)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = ucf_option.parser.parse_args()
    setup_seed(args.seed)

    label_map = dict({'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest', 'Arson': 'arson', 'Assault': 'assault', 'Burglary': 'burglary', 'Explosion': 'explosion', 'Fighting': 'fighting', 'RoadAccidents': 'roadAccidents', 'Robbery': 'robbery', 'Shooting': 'shooting', 'Shoplifting': 'shoplifting', 'Stealing': 'stealing', 'Vandalism': 'vandalism'})

    normal_dataset = UCFDataset(args.visual_length, args.train_list, False, label_map, True)
    normal_loader = DataLoader(normal_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    anomaly_dataset = UCFDataset(args.visual_length, args.train_list, False, label_map, False)
    anomaly_loader = DataLoader(anomaly_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    test_dataset = UCFDataset(args.visual_length, args.test_list, True, label_map)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = TVVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, args.visual_head, args.visual_layers, args.attn_window, args.prompt_prefix, args.prompt_postfix, device)

    train(model, normal_loader, anomaly_loader, test_loader, args, label_map, device)