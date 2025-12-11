import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops import rearrange

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

class FeedForwardModule(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super(FeedForwardModule, self).__init__()
        self.ffm = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ffm(x)


class ConformerConvModule(nn.Module):
    def __init__(self, dim, expansion_factor=2, kernel_size=31, dilation=1, dropout=0.):
        super(ConformerConvModule, self).__init__()
        inner_dim = dim * expansion_factor
        self.ccm = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n c -> b c n'),
            nn.Conv1d(dim, inner_dim*2, 1),
            nn.GLU(dim=1),
            nn.Conv1d(inner_dim, inner_dim, kernel_size=kernel_size, dilation=dilation,
                      padding=get_padding(kernel_size, dilation), groups=inner_dim), # DepthWiseConv1d 
            nn.BatchNorm1d(inner_dim),
            nn.SiLU(),
            nn.Conv1d(inner_dim, dim, 1),
            Rearrange('b c n -> b n c'),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ccm(x)


class AttentionModule(nn.Module):
    def __init__(self, dim, n_head=8, dropout=0.):
        super(AttentionModule, self).__init__()
        self.attn = nn.MultiheadAttention(dim, n_head, dropout=dropout, batch_first=True)
        self.layernorm = nn.LayerNorm(dim)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        x = self.layernorm(x)
        x, _ = self.attn(x, x, x, 
                         attn_mask=attn_mask,
                         key_padding_mask=key_padding_mask)
        return x

class Cross_MultiAttention(nn.Module):
    def __init__(self, in_channels, emb_dim, num_heads, att_dropout=0.0, aropout=0.0):
        super(Cross_MultiAttention, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.scale = emb_dim ** -0.5

        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"
        self.depth = emb_dim // num_heads


        self.proj_in = nn.Conv1d(in_channels, emb_dim, kernel_size=1, stride=1, padding=0)

        self.Wq = nn.Linear(emb_dim, emb_dim)
        self.Wk = nn.Linear(emb_dim, emb_dim)
        self.Wv = nn.Linear(emb_dim, emb_dim)

        self.proj_out = nn.Conv1d(emb_dim, in_channels, kernel_size=1, stride=1, padding=0)


    def forward(self, x, context, pad_mask=None):
        '''

        :param x: [batch_size, c, h, w]
        :param context: [batch_szie, seq_len, emb_dim]
        :param pad_mask: [batch_size, seq_len, seq_len]
        :return:
        '''
        x = x.permute(0,2,1)
        batch_size, c, t = x.shape

        x = self.proj_in(x)   # [batch_size, c, h, w] = [3, 512, 512, 512]
        x = rearrange(x, 'b c t -> b t c')   # [batch_size, h*w, c] = [3, 262144, 512]

        Q = self.Wq(x)  # [batch_size, h*w, emb_dim] = [3, 262144, 512]
        K = self.Wk(context)  # [batch_szie, seq_len, emb_dim] = [3, 5, 512]
        V = self.Wv(context)

        Q = Q.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)  # [batch_size, num_heads, h*w, depth]
        K = K.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)  # [batch_size, num_heads, seq_len, depth]
        V = V.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)

        # [batch_size, num_heads, h*w, seq_len]
        att_weights = torch.einsum('bnid,bnjd -> bnij', Q, K)
        att_weights = att_weights * self.scale

        if pad_mask is not None:
            # 因为是多头，所以mask矩阵维度要扩充到4维  [batch_size, h*w, seq_len] -> [batch_size, nums_head, h*w, seq_len]
            pad_mask = pad_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            att_weights = att_weights.masked_fill(pad_mask, -1e9)

        att_weights = F.softmax(att_weights, dim=-1)
        out = torch.einsum('bnij, bnjd -> bnid', att_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.emb_dim)   # [batch_size, h*w, emb_dim]
        out = out.permute(0,2,1)

        out = self.proj_out(out)   # [batch_size, c, h, w]
        out = out.permute(0,2,1)

        return out

class ConformerBlock(nn.Module):
    def __init__(self, dim, n_head=8, ffm_mult=4, ccm_expansion_factor=2, ccm_kernel_size=31, 
                 dilation=1, ffm_dropout=0.2, attn_dropout=0., ccm_dropout=0.2):
        super(ConformerBlock, self).__init__()
        self.ffm1 = FeedForwardModule(dim, ffm_mult, dropout=ffm_dropout)
        self.attn = AttentionModule(dim, n_head, dropout=attn_dropout)
        self.ccm = ConformerConvModule(dim, ccm_expansion_factor, ccm_kernel_size, dilation, dropout=ccm_dropout)
        self.ffm2 = FeedForwardModule(dim, ffm_mult, dropout=ffm_dropout)
        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + 0.5 * self.ffm1(x)
        x = x + self.attn(x)
        x = x + self.ccm(x)
        x = x + 0.5 * self.ffm2(x)
        x = self.post_norm(x)
        return x

class ConformerBlock_Cross(nn.Module):
    def __init__(self, dim, n_head=8, ffm_mult=4, ccm_expansion_factor=2, ccm_kernel_size=31, 
                 dilation=1, ffm_dropout=0.2, attn_dropout=0., ccm_dropout=0.2):
        super(ConformerBlock_Cross, self).__init__()
        self.ffm1 = FeedForwardModule(dim, ffm_mult, dropout=ffm_dropout)
        self.attn = Cross_MultiAttention(dim, dim, n_head)
        self.ccm = ConformerConvModule(dim, ccm_expansion_factor, ccm_kernel_size, dilation, dropout=ccm_dropout)
        self.ffm2 = FeedForwardModule(dim, ffm_mult, dropout=ffm_dropout)
        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, context):
        x = x + 0.5 * self.ffm1(x)
        x = x + self.attn(x, context)
        x = x + self.ccm(x)
        x = x + 0.5 * self.ffm2(x)
        x = self.post_norm(x)
        return x

def main():
    x = torch.randn(4, 161, 256)
    model = ConformerBlock(dim=256)
    x = model(x)
    print(x.size())

if __name__ == "__main__":
    main()