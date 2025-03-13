import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, dim, batch,vetex, heads = 4, dim_head = 16, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.down_3_pool = nn.AvgPool2d(kernel_size=3, stride=3)
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim , bias = False)
        # self.to_k = nn.Linear(dim, inner_dim , bias = False)
        # self.to_v = nn.Linear(dim, inner_dim , bias = False)
        # 可学习的
        self.to_k = nn.Parameter(torch.randn(batch, heads,vetex, dim_head))
        # self.to_v = nn.Parameter(torch.randn(batch, heads,vetex, dim_head))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # 防止内存爆炸
        print(f"最初{x.shape}")
        B1,C1,H1,W1 = x.shape
        x = self.down_3_pool(x)
        B2,C2,H2,W2 = x.shape
        # print(x.shape)
        x = x.view(B2, C2, -1).transpose(1, 2)
        print(x.shape)
        # exit(0)
        B, L, N = x.shape
        x = self.norm(x)
        print(f'x{x.shape}')
        q = self.to_q(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)
        k = self.to_k
        v = self.to_k
        # q = torch.rand((2, 8, 2,64))
        # k = torch.rand((2, 8, 2,64))
        # v = torch.rand((2, 8, 2,64))
        # x_flattened = x.view(B, -1)
        # k = linear_layer(x_flattened).view(B, , -1)
        # v = 
        print(f'q{q.shape}')
        # exit(0)
        print(f'k{k.shape}')
        print(f'v{v.shape}')
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        print(f'dots{dots.shape}')
        attn = self.attend(dots)
        attn = self.dropout(attn)
        print(f'attn{attn.shape}')

        out = torch.matmul(attn, v)
        print(f'out{out.shape}')
        out = rearrange(out, 'b h n d -> b n (h d)')
        # out
        self.to_out(out)
        # 变回2维，使用线性插值，进行放大
        out = out[:B2]
        out = out.transpose(1,2).view(B2,C2,H2,W2)
        out = F.interpolate(out, size=(H1, W1), mode='bilinear', align_corners=False)
        print(f'out2-{out.shape}')
        return out

def main():
    # 设定参数
    dim = 64  # 输入维度大小
    heads = 4  # 注意力头的数量
    dim_head = 16  # 每个头部的维度大小
    dropout = 0.1  # dropout比率

    # 实例化注意力机制层
    att_layer = Attention(dim=dim,vetex=2,batch=2, heads=heads, dim_head=dim_head, dropout=dropout)

    # 创建一个随机输入张量，形状为(batch_size, seq_len, dim)
    x = torch.rand((1, 64, 60,60))  # 示例中的batch_size为2，seq_len为32

    # 使用注意力层处理输入张量
    out = att_layer(x)

    print("输出张量形状: ", out.shape)
    # print("输出张量: ", out)

if __name__ == "__main__":
    main()