import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity

class ConvModel(nn.Module):
    def __init__(self, H, W, in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.stride = stride
        self.padding = padding

        self.H = H
        self.W = W

        # TO DO: Define static shapes here. 
        self.tile_m = 128
        self.tile_n = 32
        self.tile_k = 64
        
        # Precompute output size
        self.out_h = (H+2*padding-kernel_size)//stride+1
        self.out_w = (W+2*padding-kernel_size)//stride+1
        

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))

        

    def im2col_manual(self, x):
        N = x.shape[0]        # batch size can remain dynamic
        C = self.in_channels
        KH = KW = self.kernel_size
        S = self.stride
        P = self.padding
        out_h = self.out_h
        out_w = self.out_w

        # Pad input
        x_pad = F.pad(x, (P, P, P, P))

        # TO DO: Convert input (x) into shape (N, out_h*out_w, C*KH*KW). 
        # Refer to Lecture 3 for implementing this operation.
        patches=[]
        for kh in range(KH):
            for kw in range(KW):
                patch = x_pad[:,:,kh:kh + out_h * S:S,kw:kw + out_w * S:S]
                patches.append(patch)
        
        patches = torch.stack(patches, dim=2).reshape(N, C * KH * KW, out_h * out_w).permute(0, 2, 1).contiguous()
        return patches

    def conv2d_manual(self, x):
        N = x.shape[0]
        C_out = self.out_channels
        KH = KW = self.kernel_size

        # TO DO: 1) convert input (x) into shape (N, out_h*out_w, C*KH*KW).
        cols = self.im2col_manual(x)          

        # TO DO: 2) flatten self.weight into shape (C_out, C*KH*KW).
        K = self.in_channels * KH * KW
        weight_flat = self.weight.reshape(C_out, K)
        # TO DO: 3) perform tiled matmul after required reshaping is done.
        L = self.out_h * self.out_w
        out = cols.new_zeros((N, L, C_out))
        for i in range(0, L, self.tile_m):
            i_end = min(i + self.tile_m, L)
            for j in range(0, C_out, self.tile_n):
                j_end = min(j + self.tile_n, C_out)
                acc = cols.new_zeros((N, i_end - i, j_end - j))
                for k in range(0, K, self.tile_k):
                    k_end = min(k + self.tile_k, K)
                    acc = acc + torch.matmul(cols[:, i:i_end, k:k_end],weight_flat[j:j_end, k:k_end].T)
                out[:, i:i_end, j:j_end] = acc
        # TO DO: 4) Add bias.
        out = out + self.bias.reshape(1, 1, C_out)

        # TO DO: 5) reshape output into shape (N, C_out, out_h, out_w).
        out = out.permute(0, 2, 1).reshape(N, C_out, self.out_h, self.out_w )
        return out

    def forward(self, x):
        return self.conv2d_manual(x)


if __name__ == "__main__":
    torch.manual_seed(0)
    N, C, H, W = 2, 4, 22, 22
    x = torch.randn(N, C, H, W, device="cuda")
    out_channels=8
    kernel_size=7
    model = ConvModel(H, W, C, out_channels, kernel_size, stride=1, padding=1).cuda()
    out = model(x)

    # Test your solution
    conv_ref = F.conv2d(x, model.weight, model.bias, stride=1, padding=1)
    print("PyTorch --- shape check:", out.shape == conv_ref.shape)
    print("PyTorch --- correctness check:", torch.allclose(out, conv_ref, atol=1e-4))
