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

class NoisyBranch(torch.nn.Module):
    def __init__(self, input_dim, project_dim, output_dim, num_blocks=8):
        super().__init__()
        self.num_blocks = num_blocks
        self.input_project = nn.Linear(input_dim, project_dim)
        self.Conformer = nn.ModuleList([])
        for i in range(num_blocks):
            self.Conformer.append(ConformerBlock(project_dim, dilation=2 ** (i % 4)))
        self.norm = nn.LayerNorm(project_dim, eps=1e-6)
        self.output_project = nn.Linear(project_dim, output_dim)
    
    def forward(self, x):
        x = self.input_project(x)
        for i in range(self.num_blocks):
            x = self.Conformer[i](x) + x
            x = self.norm(x)
        x = self.output_project(x)
        return x

class TokenGenerator(nn.Module):
    def __init__(self, code_size=1024, mid_dim=512, num_blocks=4):
        super().__init__()
        self.num_blocks = num_blocks
        self.feature_project = nn.Linear(1024, mid_dim)
        self.project = nn.Linear(2*mid_dim, mid_dim)
        self.Conformer = nn.ModuleList([])
        for i in range(num_blocks):
            self.Conformer.append(ConformerBlock(mid_dim, dilation=2 ** (i % 4)))
        self.norm = nn.LayerNorm(mid_dim, eps=1e-6)
        self.output_project = nn.Linear(mid_dim, code_size)

    def forward(self, x, dac):
        x = self.feature_project(x)
        x = torch.cat((x, dac),-1)
        x = self.project(x)
        for i in range(self.num_blocks):
            x = self.Conformer[i](x) + x
            x = self.norm(x)
        x = self.output_project(x)
        
        return x

class UDSE(nn.Module):
    def __init__(self, code_size=1024, project_dim=512, mid_dim=512, num_quantize=9, code_dim=8):
        super().__init__()
        self.num_quantize = num_quantize
        self.branch = NoisyBranch(input_dim=num_quantize*code_dim, project_dim=project_dim, output_dim=mid_dim, num_blocks=8)
        self.backbone = nn.ModuleList([])
        for i in range(num_quantize):
            self.backbone.append(TokenGenerator(code_size=code_size, mid_dim=mid_dim, num_blocks=4))
    
    def forward(self, x, dac):
        quan_list = []
        dac = self.branch(dac)
        for i in range(self.num_quantize):
            y = self.backbone[i](x[i], dac)
            quan_list.append(y)
        return quan_list

class infer_UDSE(nn.Module):
    def __init__(self, code_size=1024, project_dim=512, mid_dim=512, num_quantize=9, code_dim=8):
        super().__init__()
        self.num_quantize = num_quantize
        self.branch = NoisyBranch(input_dim=num_quantize*code_dim, project_dim=project_dim, output_dim=mid_dim, num_blocks=8)
        self.backbone = nn.ModuleList([])
        for i in range(num_quantize):
            self.backbone.append(TokenGenerator(code_size=code_size, mid_dim=mid_dim, num_blocks=4))

    def forward(self, x, dac, dac_model):
        quan_list = []
        dac = self.branch(dac)
        input_emb = x

        for i in range(self.num_quantize):
            logits = self.backbone[i](input_emb, dac)
            logits = F.softmax(logits, dim=-1)
            codes = torch.argmax(logits, dim=-1).unsqueeze(2).permute(0, 2, 1)
            quan_list.append(codes)

            if i < self.num_quantize - 1:  
                input_token = torch.cat(quan_list, dim=1)
                input_emb, _, _ = dac_model.quantizer.from_codes(input_token)
                input_emb = input_emb.permute(0, 2, 1)
        
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