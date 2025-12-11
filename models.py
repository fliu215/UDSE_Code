import torch
import dac
import sys
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, spectral_norm
from utils import init_weights, get_padding
import numpy as np
from Conformer import ConformerBlock

def log(t, eps = 1e-20):
    return torch.log(t + eps)

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim = dim)

def top_k(prob, k=512):
    k = k
    top_k_values, top_k_indices = torch.topk(prob, k, dim=-1)
    top_k_logits = torch.full_like(prob, float('-inf'))
    top_k_logits = top_k_logits.scatter_(-1, top_k_indices, top_k_values)
    token_g = gumbel_sample(top_k_logits, temperature = 0.1, dim = -1).unsqueeze(2)
    return token_g

class NoisyBranch(torch.nn.Module):
    def __init__(self, input_dim, project_dim, output_dim, num_blocks=1):
        super().__init__()
        self.num_blocks = num_blocks
        self.input_project = nn.Linear(input_dim, project_dim)
        self.DFconformer = nn.ModuleList([])
        for i in range(num_blocks):
            self.DFconformer.append(ConformerBlock(project_dim, dilation=2 ** (i % 4)))
        self.norm = nn.LayerNorm(project_dim, eps=1e-6)
        self.output_project = nn.Linear(project_dim, output_dim)
    
    def forward(self, x):
        x = self.input_project(x)
        for i in range(self.num_blocks):
            x = self.DFconformer[i](x) + x
            x = self.norm(x)
        x = self.output_project(x)
        return x

class TokenGenerator(nn.Module):
    def __init__(self, code_size=1024, mid_dim=512, num_blocks=1):
        super().__init__()
        self.num_blocks = num_blocks
        self.feature_project = nn.Linear(1024, mid_dim)
        # self.branch = NoisyBranch(input_dim=num_quantize*code_dim, project_dim=project_dim, output_dim=mid_dim)
        self.project = nn.Linear(2*mid_dim, mid_dim)
        self.DFconformer = nn.ModuleList([])
        for i in range(num_blocks):
            self.DFconformer.append(ConformerBlock(mid_dim, dilation=2 ** (i % 4)))
        self.norm = nn.LayerNorm(mid_dim, eps=1e-6)
        self.output_project = nn.Linear(mid_dim, code_size)

    def forward(self, x, dac):
        x = self.feature_project(x)
        # cond0 = self.branch(dac)
        # cond1 = self.feature_project(cond1)
        # cond = torch.cat((dac, cond1), dim=-1)
        # cond = self.project(cond)
        x = torch.cat((x, dac),-1)
        x = self.project(x)
        for i in range(self.num_blocks):
            x = self.DFconformer[i](x) + x
            x = self.norm(x)
        x = self.output_project(x)
        
        return x

class Parallel(nn.Module):
    def __init__(self, project_dim=256, mid_dim=256, num_quantize=9, code_dim=8):
        super().__init__()
        self.num_quantize = num_quantize
        self.branch = NoisyBranch(input_dim=num_quantize*code_dim, project_dim=project_dim, output_dim=mid_dim)
        self.backbone = nn.ModuleList([])
        for i in range(num_quantize):
            self.backbone.append(TokenGenerator(mid_dim=mid_dim))
    
    def forward(self, x, dac):
        quan_list = []
        dac = self.branch(dac)
        for i in range(self.num_quantize):
            y = self.backbone[i](x[i], dac)
            quan_list.append(y)
        return quan_list

class infer_Parallel(nn.Module):
    def __init__(self, project_dim=256, mid_dim=256, num_quantize=9, code_dim=8):
        super().__init__()
        self.num_quantize = num_quantize
        self.branch = NoisyBranch(input_dim=num_quantize*code_dim, project_dim=project_dim, output_dim=mid_dim)
        self.backbone = nn.ModuleList([])
        for i in range(num_quantize):
            self.backbone.append(TokenGenerator(mid_dim=mid_dim))

    def forward(self, x, dac, dac_model):
        quan_list = []
        dac = self.branch(dac)
        x1 = self.backbone[0](x, dac)
        x1 = F.softmax(x1, dim=-1)
        x1 = torch.argmax(x1, dim=-1).unsqueeze(2).permute(0,2,1)
        quan_list.append(x1)
        x1_emb, _, _ = dac_model.quantizer.from_codes(x1)
        x1_emb = x1_emb.permute(0,2,1)

        x2 = self.backbone[1](x1_emb, dac)
        x2 = F.softmax(x2, dim=-1)
        x2 = torch.argmax(x2, dim=-1).unsqueeze(2).permute(0,2,1)
        quan_list.append(x2)
        input_token = torch.cat((x1,x2),dim=1)
        x2_emb, _, _ = dac_model.quantizer.from_codes(input_token)
        x2_emb = x2_emb.permute(0,2,1)

        x3 = self.backbone[2](x2_emb, dac)
        x3 = F.softmax(x3, dim=-1)
        x3 = torch.argmax(x3, dim=-1).unsqueeze(2).permute(0,2,1)
        quan_list.append(x3)
        input_token = torch.cat((x1,x2,x3),dim=1)
        x3_emb, _, _ = dac_model.quantizer.from_codes(input_token)
        x3_emb = x3_emb.permute(0,2,1)

        x4 = self.backbone[3](x3_emb, dac)
        x4 = F.softmax(x4, dim=-1)
        x4 = torch.argmax(x4, dim=-1).unsqueeze(2).permute(0,2,1)
        quan_list.append(x4)
        input_token = torch.cat((x1,x2,x3,x4),dim=1)
        x4_emb, _, _ = dac_model.quantizer.from_codes(input_token)
        x4_emb = x4_emb.permute(0,2,1)

        x5 = self.backbone[4](x4_emb, dac)
        x5 = F.softmax(x5, dim=-1)
        x5 = torch.argmax(x5, dim=-1).unsqueeze(2).permute(0,2,1)
        quan_list.append(x5)
        input_token = torch.cat((x1,x2,x3,x4,x5),dim=1)
        x5_emb, _, _ = dac_model.quantizer.from_codes(input_token)
        x5_emb = x5_emb.permute(0,2,1)

        x6 = self.backbone[5](x5_emb, dac)
        x6 = F.softmax(x6, dim=-1)
        x6 = torch.argmax(x6, dim=-1).unsqueeze(2).permute(0,2,1)
        quan_list.append(x6)
        input_token = torch.cat((x1,x2,x3,x4,x5,x6),dim=1)
        x6_emb, _, _ = dac_model.quantizer.from_codes(input_token)
        x6_emb = x6_emb.permute(0,2,1)

        x7 = self.backbone[6](x6_emb, dac)
        x7 = F.softmax(x7, dim=-1)
        x7 = torch.argmax(x7, dim=-1).unsqueeze(2).permute(0,2,1)
        quan_list.append(x7)
        input_token = torch.cat((x1,x2,x3,x4,x5,x6,x7),dim=1)
        x7_emb, _, _ = dac_model.quantizer.from_codes(input_token)
        x7_emb = x7_emb.permute(0,2,1)

        x8 = self.backbone[7](x7_emb, dac)
        x8 = F.softmax(x8, dim=-1)
        x8 = torch.argmax(x8, dim=-1).unsqueeze(2).permute(0,2,1)
        quan_list.append(x8)
        input_token = torch.cat((x1,x2,x3,x4,x5,x6,x7,x8),dim=1)
        x8_emb, _, _ = dac_model.quantizer.from_codes(input_token)
        x8_emb = x8_emb.permute(0,2,1)
        
        x9 = self.backbone[8](x8_emb, dac)
        x9 = F.softmax(x9, dim=-1)
        x9 = torch.argmax(x9, dim=-1).unsqueeze(2).permute(0,2,1)
        quan_list.append(x9)
        return torch.cat(quan_list, dim=1)


def main():
    x = []
    for i in range(9):
        y = torch.randint(low=0, high=1023, size=(4,161,1))
        x.append(y)
    cond1 = torch.randn(4,161,1024)
    dac = torch.randn(4,161,9*8)
    model = Parallel()
    x = model(x, dac, cond1)
    print(x[0].size())

if __name__ == "__main__":
    main()