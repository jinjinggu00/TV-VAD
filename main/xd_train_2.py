import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import random

from model import TVVAD
from xd_test import test
from utils.dataset import XDDataset
from utils.tools import get_prompt_text, get_batch_label
import xd_option
from torch import nn


def symmetric_kl_divergence(p, q):
    """
    Compute the Symmetric KL Divergence between two probability distributions.

    Args:
        p (torch.Tensor): Probability distribution P. Shape (N, D), where N is the batch size and D is the distribution dimension.
        q (torch.Tensor): Probability distribution Q. Shape (N, D), where N is the batch size and D is the distribution dimension.

    Returns:
        torch.Tensor: The symmetric KL divergence bet ween P and Q.
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

def kl_loss(A, B):
    return nn.KLDivLoss(reduction='batchmean')(torch.log_softmax(B, dim=1), torch.softmax(A,dim=1))

def CLASM(logits,labels, lengths, device):  # 96*256*7
    instance_logits = torch.zeros(0).to(device)
    labels = labels / torch.sum(labels, dim=1, keepdim=True)

    labels = labels.to(device)



    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True, dim=0)  # torch.Size([shi ji , 7])
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)


    milloss = -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)    # instance_logits ---> torch.Size([96, 7])  # labels ---> torch.Size([96, 7])

    soft_instance_logist = F.softmax(instance_logits, dim=-1)

    position_loss = F.cross_entropy(soft_instance_logist, labels.argmax(dim = -1))

    p = soft_instance_logist
    q = labels+0.0002


    kl_loss = symmetric_kl_divergence(p, q)

    return milloss , position_loss, kl_loss


def CLAS2(logits, labels, lengths, device):

    instance_logits = torch.zeros(0).to(device)
    labels = 1 - labels[:, 0].reshape(labels.shape[0])
    labels = labels.to(device)
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True)
        tmp = torch.mean(tmp).view(1)
        instance_logits = torch.cat((instance_logits, tmp))


    clsloss = F.binary_cross_entropy(instance_logits, labels)

    return clsloss




def train(model, train_loader, test_loader, args, label_map: dict, device):
    model.to(device)

    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
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
        for i, item in enumerate(train_loader):
            step = 0
            visual_feat, text_labels_, feat_lengths = item
            visual_feat = visual_feat.to(device)
            feat_lengths = feat_lengths.to(device)


            text_labels = get_batch_label(text_labels_, prompt_text, label_map).to(device)


            text_features, logits1, logits2 = model(visual_feat, None, prompt_text, feat_lengths)  # torch.Size([7, 512]) torch.Size([96, 256, 1]) torch.Size([96, 256, 7])


            loss1 = CLAS2(logits1, text_labels, feat_lengths, device)
            loss_total1 += loss1.item()

            loss2, position_loss, kl = CLASM(logits2, text_labels, feat_lengths, device)



            loss_total2 += loss2.item()
            posi_loss_total += position_loss.item()
            kl_total += kl.item()

            loss3 = torch.zeros(1).to(device)
            text_feature_normal = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)  # torch.Size([512])

            for j in range(1, text_features.shape[0]):
                text_feature_abr = text_features[j] / text_features[j].norm(dim=-1, keepdim=True)
                loss3 += torch.abs(text_feature_normal @ text_feature_abr)
            loss3 = loss3 / 6

            loss = loss1 *1.2 + loss2*1   + position_loss + kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += i * train_loader.batch_size
            if step % 4800 == 0 and step != 0:
                print('epoch: ', e+1, '| step: ', step, '| loss1: ', loss_total1 / (i+1), '| loss2: ', loss_total2 / (i+1), '| loss3: ', loss3.item(), '| kl : ', kl_total/(i+1))
                
        scheduler.step()
        flage = False
        AUC, AP, mAP = test(model, test_loader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)

        if AP > ap_best:
            ap_best = AP 
            checkpoint = {
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ap': ap_best}
            torch.save(checkpoint, args.checkpoint_path)

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
    args = xd_option.parser.parse_args()
    setup_seed(args.seed)

    label_map = dict({'A': 'normal', 'B1': 'fighting', 'B2': 'shooting', 'B4': 'riot', 'B5': 'abuse', 'B6': 'car accident', 'G': 'explosion'})

    train_dataset = XDDataset(args.visual_length, args.train_list, False, label_map)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    test_dataset = XDDataset(args.visual_length, args.test_list, True, label_map)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True)

    model = TVVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, args.visual_head, args.visual_layers, args.attn_window, args.prompt_prefix, args.prompt_postfix, device)
    train(model, train_loader, test_loader, args, label_map, device)