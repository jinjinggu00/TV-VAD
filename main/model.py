from collections import OrderedDict
import torch.fft
import numpy as np
import torch
import torch.nn.functional as F


from torch import nn
from clip import clip
from utils.layers import GraphConvolution, DistanceAdj
from translayer import *

import torch
import numpy as np
import matplotlib.pyplot as plt
import math

# 能量集中度计算函数
def energy_concentration(x, top_k_ratio=0.1):
    """
    计算频域能量集中度
    :param x: 输入特征 (B, H, W, C) 或 (B, N, C)，需为实数或复数
    :param top_k_ratio: 前 k 比例能量分量 (默认取前 10%)
    :return: 能量集中度
    """
    # 计算频域特征 (2D FFT)
    x_fft = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")  # 对 H, W 维度做傅里叶变换
    energy = torch.abs(x_fft) ** 2  # 计算频率分量的能量 |F_i|^2

    # 展平频域特征，按能量降序排序
    energy_flat = energy.view(x.shape[0], -1)  # 展平成 (B, N)
    sorted_energy, _ = torch.sort(energy_flat, dim=-1, descending=True)

    # 计算总能量和累积能量
    total_energy = torch.sum(sorted_energy, dim=-1)  # 总能量 (B,)
    top_k = int(top_k_ratio * sorted_energy.shape[1])  # 选取前 k 分量
    top_k_energy = torch.sum(sorted_energy[:, :top_k], dim=-1)  # 前 k 能量总和

    # 能量集中度
    concentration = top_k_energy / total_energy  # (B,)
    return concentration

def compute_sparsity(fft_features):
    """
    计算频域特征的稀疏性
    """
    # 转换为幅值
    spectrum = torch.abs(fft_features)
    total_elements = spectrum.numel()
    non_zero_elements = torch.sum(spectrum > 1e-3).item()

    sparsity = 1 - (non_zero_elements / total_elements)
    return sparsity


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()

        # 定义第一层
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()

        # 定义第二层
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()

        # 定义输出层
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 前向传播
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out


class F3R(nn.Module):
    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1, hidden_size_factor=1):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def forward(self, x, spatial_size=None):
        bias = x

        dtype = x.dtype
        x = x.float()
        B, N, C = x.shape

        if spatial_size == None:
            H = W = int(math.sqrt(N))
        else:
            H, W = spatial_size

        x = x.reshape(B, H, W, C)
        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho") #



        x = x.reshape(B, x.shape[1], x.shape[2], self.num_blocks, self.block_size)

        o1_real = torch.zeros([B, x.shape[1], x.shape[2], self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o1_imag = torch.zeros([B, x.shape[1], x.shape[2], self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        total_modes = N // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        o1_real[:, :, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].real, self.w1[0]) - \
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag[:, :, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].imag, self.w1[0]) + \
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].real, self.w1[1]) + \
            self.b1[1]
        )

        o2_real[:, :, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_real[:, :, :kept_modes], self.w2[0]) - \
            torch.einsum('...bi,bio->...bo', o1_imag[:, :, :kept_modes], self.w2[1]) + \
            self.b2[0]
        )

        o2_imag[:, :, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_imag[:, :, :kept_modes], self.w2[0]) + \
            torch.einsum('...bi,bio->...bo', o1_real[:, :, :kept_modes], self.w2[1]) + \
            self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold) #


        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], x.shape[2], C)  # torch.Size([128, 16, 9, 512])

        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm="ortho") #
        x = x.reshape(B, N, C)
        x = x.type(dtype)   # torch.Size([128, 256, 512])
        return x + bias




class LayerNorm(nn.LayerNorm):

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, padding_mask: torch.Tensor):
        padding_mask = padding_mask.to(dtype=bool, device=x.device) if padding_mask is not None else None
        self.attn_mask = self.attn_mask.to(device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, key_padding_mask=padding_mask, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        x, padding_mask = x
        x = x + self.attention(self.ln_1(x), padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, padding_mask)


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class TVVAD(nn.Module):
    def __init__(self,
                 num_class: int,
                 embed_dim: int,
                 visual_length: int,
                 visual_width: int,
                 visual_head: int,
                 visual_layers: int,
                 attn_window: int,
                 prompt_prefix: int,
                 prompt_postfix: int,
                 device):
        super().__init__()

        self.num_class = num_class
        self.visual_length = visual_length
        self.visual_width = visual_width
        self.embed_dim = embed_dim
        self.attn_window = attn_window
        self.prompt_prefix = prompt_prefix
        self.prompt_postfix = prompt_postfix
        self.device = device

        self.temporal = Transformer(
            width=visual_width,
            layers=visual_layers,
            heads=visual_head,
            attn_mask=self.build_attention_mask(self.attn_window)
        )

        width = int(visual_width / 2)

        self.gc1 = GraphConvolution(visual_width, width, residual=True)
        self.gc2 = GraphConvolution(width, width, residual=True)
        self.gc3 = GraphConvolution(visual_width, width, residual=True)
        self.gc4 = GraphConvolution(width, width, residual=True)


        self.disAdj = DistanceAdj()
        self.linear = nn.Linear(visual_width, visual_width)
        self.gelu = QuickGELU()

        self.mlp1 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        self.mlp2 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        self.classifier = nn.Linear(visual_width, 1)

        self.clipmodel, _ = clip.load("ViT-B/16", device)
        for clip_param in self.clipmodel.parameters():
            clip_param.requires_grad = False

        self.frame_position_embeddings = nn.Embedding(visual_length, visual_width)
        self.text_prompt_embeddings = nn.Embedding(77, self.embed_dim)

        self.initialize_parameters()



        # 9-12
        self.F3R_ = F3R(512)

        self.self_att = Transformer_zhang(512, 2, 4, 128, 512, dropout=0.5).to(device)



    def initialize_parameters(self):
        nn.init.normal_(self.text_prompt_embeddings.weight, std=0.01)
        nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)

    def build_attention_mask(self, attn_window):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.visual_length, self.visual_length)
        mask.fill_(float('-inf'))
        for i in range(int(self.visual_length / attn_window)):
            if (i + 1) * attn_window < self.visual_length:
                mask[i * attn_window: (i + 1) * attn_window, i * attn_window: (i + 1) * attn_window] = 0
            else:
                mask[i * attn_window: self.visual_length, i * attn_window: self.visual_length] = 0

        return mask

    def adj4(self, x, seq_len):
        soft = nn.Softmax(1)
        x2 = x.matmul(x.permute(0, 2, 1)) # B*T*T
        x_norm = torch.norm(x, p=2, dim=2, keepdim=True)  # B*T*1
        x_norm_x = x_norm.matmul(x_norm.permute(0, 2, 1))
        x2 = x2/(x_norm_x+1e-20)
        output = torch.zeros_like(x2)
        if seq_len is None:
            for i in range(x.shape[0]):
                tmp = x2[i]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i] = adj2
        else:
            for i in range(len(seq_len)):
                tmp = x2[i, :seq_len[i], :seq_len[i]]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i, :seq_len[i], :seq_len[i]] = adj2

        return output


    def get_labels_from_tuple(self, tuple_data, label_map):

        """
        将输入的元组数据进行简化处理，并根据label_map映射得到对应的标签列表。

        Args:
        - tuple_data (tuple): 输入的元组数据，每个元素需要进行简化处理。
        - label_map (dict): 标签映射字典，将简化后的元素映射为对应的标签。

        Returns:
        - labels (list): 包含标签的列表，与输入的元组长度相同。
        """

        def simplify_element(element):
            if '-' in element:
                return element.split('-')[0]  # 取第一个横杠前面的部分作为简化后的元素
            else:
                return element

        simplified_tuple = tuple(simplify_element(item) for item in tuple_data)
        labels = [label_map.get(item, 'unknown') for item in simplified_tuple]
        return labels


    def encode_video(self, images, padding_mask, lengths):
        images = images.to(torch.float)
        position_ids = torch.arange(self.visual_length, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand(images.shape[0], -1)
        frame_position_embeddings = self.frame_position_embeddings(position_ids)
        frame_position_embeddings = frame_position_embeddings.permute(1, 0, 2)
        images = images.permute(1, 0, 2) + frame_position_embeddings  # torch.Size([256, 128, 512])


        images, _ = self.temporal((images, None))  # torch.Size([256, 128, 512])  0 1 2
        x = images.permute(1, 0, 2)  # 128 256 512

        adj = self.adj4(x, lengths)  # ([128, 256, 256])
        disadj = self.disAdj(x.shape[0], x.shape[1])  # ([128, 256, 256])

        x1_h = self.gelu(self.gc1(x, adj))
        x2_h = self.gelu(self.gc3(x, disadj))

        x1 = self.gelu(self.gc2(x1_h, adj))  # torch.Size([128, 256, 256])   #H
        x2 = self.gelu(self.gc4(x2_h, disadj)) # torch.Size([128, 256, 256])   L

        x__ = torch.cat((x1, x2), 2)  # torch.Size([128, 256, 512])




        # x__ = images.permute(1, 0, 2)  # 128 256 512
        x = self.F3R_(x__)


        x = self.linear(x)  # torch.Size([128, 256, 512])


        return x

    def encode_textprompt(self, text):
        word_tokens = clip.tokenize(text).to(self.device)
        word_embedding = self.clipmodel.encode_token(word_tokens)
        text_embeddings = self.text_prompt_embeddings(torch.arange(77).to(self.device)).unsqueeze(0).repeat([len(text), 1, 1])
        text_tokens = torch.zeros(len(text), 77).to(self.device)

        for i in range(len(text)):
            ind = torch.argmax(word_tokens[i], -1)
            text_embeddings[i, 0] = word_embedding[i, 0]
            text_embeddings[i, self.prompt_prefix + 1: self.prompt_prefix + ind] = word_embedding[i, 1: ind]
            text_embeddings[i, self.prompt_prefix + ind + self.prompt_postfix] = word_embedding[i, ind]
            text_tokens[i, self.prompt_prefix + ind + self.prompt_postfix] = word_tokens[i, ind]

        text_features = self.clipmodel.encode_text(text_embeddings, text_tokens)

        return text_features




    def forward(self, visual, padding_mask, text, lengths):
        visual_features = self.encode_video(visual, padding_mask, lengths)  # torch.Size([128, 256, 512])


        # print('visual_features shape is', visual_features.shape)
        logits1 = self.classifier(visual_features + self.mlp2(visual_features))  # torch.Size([128, 256, 1])


        text_features_ori = self.encode_textprompt(text)  # torch.Size([14, 512])

        text_features = text_features_ori
        logits_attn = logits1.permute(0, 2, 1)
        visual_attn = logits_attn @ visual_features
        visual_attn = visual_attn / visual_attn.norm(dim=-1, keepdim=True)
        visual_attn = visual_attn.expand(visual_attn.shape[0], text_features_ori.shape[0],
                                         visual_attn.shape[2])  # ([128, 14, 512])

        text_features = text_features_ori.unsqueeze(0)
        text_features = text_features.expand(visual_attn.shape[0], text_features.shape[1],
                                             text_features.shape[2])  # torch.Size([128, 14, 512])

        # print('text_features shape is ', text_features.shape)

        text_features = text_features + visual_attn
        text_features = text_features + self.mlp1(text_features)

        # text_features = self.cross_att_fusion(text_features, visual_attn,visual_attn)

        visual_features_norm = visual_features / visual_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features_norm.permute(0, 2, 1)
        logits2 = visual_features_norm @ text_features_norm.type(visual_features_norm.dtype) / 0.07

        return text_features_ori, logits1, logits2
    