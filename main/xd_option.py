import argparse

parser = argparse.ArgumentParser(description='VadCLIP')
parser.add_argument('--seed', default=234, type=int)

parser.add_argument('--embed-dim', default=512, type=int)
parser.add_argument('--visual-length', default=256, type=int)
parser.add_argument('--visual-width', default=512, type=int)
parser.add_argument('--visual-head', default=1, type=int)
parser.add_argument('--visual-layers', default=1, type=int)
parser.add_argument('--attn-window', default=64, type=int)
parser.add_argument('--prompt-prefix', default=10, type=int)
parser.add_argument('--prompt-postfix', default=10, type=int)
parser.add_argument('--classes-num', default=7, type=int)

parser.add_argument('--max-epoch', default=10, type=int)
parser.add_argument('--model-path', default='/home/zhangheng/paper_code/VadCLIP-main/VadCLIP-main/model/model_xd_TCA.pth')
parser.add_argument('--use-checkpoint', default=False, type=bool)
parser.add_argument('--checkpoint-path', default='/home/zhangheng/paper_code/VadCLIP-main/VadCLIP-main/model/checkpoint_TCA.pth')
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--train-list', default='/home/zhangheng/paper_code/VadCLIP-main/VadCLIP-main/list/xd_CLIP_rgb.csv')
parser.add_argument('--test-list', default='/home/zhangheng/paper_code/VadCLIP-main/VadCLIP-main/list/xd_CLIP_rgbtest.csv')
parser.add_argument('--gt-path', default='/home/zhangheng/paper_code/VadCLIP-main/VadCLIP-main/list/gt.npy')
parser.add_argument('--gt-segment-path', default='/home/zhangheng/paper_code/VadCLIP-main/VadCLIP-main/list/gt_segment.npy')
parser.add_argument('--gt-label-path', default='/home/zhangheng/paper_code/VadCLIP-main/VadCLIP-main/list/gt_label.npy')

parser.add_argument('--lr', default=1e-5)
parser.add_argument('--scheduler-rate', default=0.1)
parser.add_argument('--scheduler-milestones', default=[3, 6, 10])