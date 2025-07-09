import torch
from torch import nn
from functools import partial
from .spade_res import *

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Attention_img(nn.Module):
    def __init__(self, dim, in_chans, q_chanel, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.img_chanel = in_chans + 1
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x,):
        x_img = x[:, :self.img_chanel, :]
        x_lm = x[:, self.img_chanel:, :]

        B, N, C = x_img.shape
        kv = self.kv(x_img).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv.unbind(0) # make torchscript happy (cannot use tensor as tuple)
        q = x_lm.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_img = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_img = self.proj(x_img)
        x_img = self.proj_drop(x_img)

        return x_img

class Cross_fusion(nn.Module):

    def __init__(self, dim, in_chans, q_chanel, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.img_chanel = in_chans + 1
        self.num_channels = in_chans + q_chanel + 2
        self.scale = nn.Parameter(torch.ones(1))
        self.attn_img = Attention_img(dim, in_chans = in_chans, q_chanel = q_chanel, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.conv = nn.Conv1d(self.num_channels, self.num_channels, 1)

    def forward(self, x):
        x_img = x[:,:self.img_chanel, :]
        x_lm = x[:,self.img_chanel:, :]

        # lm_attention
        x_img = x_img + self.drop_path(self.attn_img(self.norm1(x)))
        x_img = x_img + self.drop_path(self.mlp(self.norm2(x_img)))

        x_lm = x_lm * self.scale
        x = torch.cat((x_img, x_lm), dim=1)
        x = self.conv(x)
        return x

class Pyramid_net(nn.Module):

    def __init__(self, dim, in_chans, q_chanel, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.block_l = Cross_fusion(
                dim=dim, in_chans=in_chans, q_chanel=q_chanel, num_heads=num_heads,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,drop=drop, attn_drop=attn_drop,
                drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer)

        self.block_m = Cross_fusion(
            dim=dim//2, in_chans=in_chans, q_chanel=q_chanel, num_heads=num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
            drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer)

        self.block_s = Cross_fusion(
            dim=dim//4, in_chans=in_chans, q_chanel=q_chanel, num_heads=num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
            drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer)

        n_channels = (in_chans+1) + (q_chanel+1)

        self.upsample_m = nn.ConvTranspose1d(n_channels, n_channels, kernel_size=2, stride=2)
        self.upsample_s = nn.ConvTranspose1d(n_channels, n_channels, kernel_size=2, stride=2)


    def forward(self, x):
        x_l, x_m, x_s = x[0], x[1], x[2]

        x_l = self.block_l(x_l)
        x_m = self.block_m(x_m)
        x_s = self.block_s(x_s)

        x_m = self.upsample_s(x_s) + x_m
        x_l = x_l + self.upsample_m(x_m)
        x = [x_l, x_m, x_s]
        return x


class HyVisionTransformer(nn.Module):

    def __init__(self, in_chans=49, q_chanel = 49, num_classes=1000, embed_dim=512, depth=12,
                 num_heads=8, mlp_ratio=4., qkv_bias=True, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,  norm_layer=None,
                 act_layer=None, weight_init=''):

        super().__init__()
        # self.num_classes = num_classes
        self.in_chans = in_chans
        # self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, in_chans + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        n_channels = (in_chans+1) + (q_chanel+1)
        self.downsample_m = nn.Conv1d(n_channels, n_channels, kernel_size=2, stride=2)
        self.downsample_s = nn.Conv1d(n_channels, n_channels, kernel_size=4, stride=4)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Pyramid_net(
                dim=embed_dim, in_chans=in_chans, q_chanel=q_chanel, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)



    def forward(self, x, x_lm):
        B = x.shape[0]  # [B, 49, 512]

        xlm_cls = torch.mean(x_lm, 1).view(B, 1, -1)
        x_lm = torch.cat((xlm_cls, x_lm), dim=1)

        x_cls = torch.mean(x, 1).view(B, 1, -1) # [B, 1, 512]
        x = torch.cat((x_cls, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        new_x = torch.cat((x, x_lm), dim=1)

        ###############################

        new_x_l = new_x
        new_x_m = self.downsample_m(new_x)
        new_x_s = self.downsample_s(new_x)
        new_x_in = [new_x_l,new_x_m,new_x_s]

        #############################
        new_x_in = self.blocks(new_x_in)
        new_x_l = new_x_in[0]
        new_x_l = self.norm(new_x_l)
        x_class1 = new_x_l[:,0,:]

        return x_class1




