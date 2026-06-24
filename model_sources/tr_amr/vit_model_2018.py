"""
original code from rwightman:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
from functools import partial
from collections import OrderedDict
import math

import torch
import torch.nn as nn


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize,这个floor方法就是向下取整1.99输出也是1
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=1024, patch_size=(2, 32), in_c=1, embed_dim=64, norm_layer=None):
        super().__init__()
        img_size = (2, img_size)  # 2*1024
        # patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        # self.grid_size = (2, 2)  # grid的大小是2*2
        # self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.stride = (1, 32)
        self.grid_size = (
            (img_size[0] - patch_size[0]) // self.stride[0] + 1,
            (img_size[1] - patch_size[1]) // self.stride[1] + 1,
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        # self.num_patches = self.grid_size[0] * self.grid_size[1]

        # self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=self.stride)
        '''
        使用卷积的方法来生成Patch，输入的通道数是3,输出的通道数是768，这个768是因为
        ViT-B/16也就是ViT-Base patch大小是16*16的模型中使用的d_model就是768，
        卷积核的大小是patch的size，步长也是patch的size，这样的话就能够完美的把图像
        分割为不同的patch了
        '''
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        # LN归一化我之前还真的没接触过，这个LN传入的参数居然就是只有参数的维度嘛
        '''
        如果传入整数，比如4，则被看做只有一个整数的list，此时LayerNorm会对输入的最后
        一维进行归一化，这个int值需要和输入的最后一维一样大。假设此时输入的数据维度是
        [3, 4]，则对3个长度为4的向量求均值方差，得到3个均值和3个方差，分别对这3行进行
        归一化（每一行的4个数字都是均值为0，方差为1）；LayerNorm中的weight和bias也
        分别包含4个数字，重复使用3次，对每一行进行仿射变换（仿射变换即乘以weight中对
        应的数字后，然后加bias中对应的数字），并会在反向传播时得到学习。
        '''
        # 也就是在使用LN归一化的时候，只需要传入指定normalized_shape就可以了。
        # 比如这个传入了768，会对每个768的patch embedding进行LN归一化

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x  # [B,HW,C]
        # 产生最终的patch embedding

class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 # 这个vit-b/16的dim应该是768
                 num_heads=16,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads  # 所有的头的维度相加应该为总的dim
        self.scale = qk_scale or head_dim ** -0.5  # qk矩阵的点乘结果应该除以的一个系数
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # 同时生成qkc矩阵
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)  # 得到softmax的输出结果
        attn = self.attn_drop(attn)
        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)  # 需要这么多的dropout嘛

        return x

class SquaredReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = self.relu(x)
        return x * x

class GAU(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 # 这个vit-b/16的dim应该是768
                 num_heads=16,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(GAU, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads  # 所有的头的维度相加应该为总的dim
        self.scale = qk_scale or head_dim ** -0.5  # qk矩阵的点乘结果应该除以的一个系数
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # 同时生成qkc矩阵
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        self.squ_relu = SquaredReLU()

    def forward(self, x):
        input = x
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)  # 得到softmax的输出结果
        attn = self.attn_drop(attn)
        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)  # 需要这么多的dropout嘛

        return input + x

class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features  # 由于out_features未传入，所以为in_features=32
        hidden_features = hidden_features or in_features
        # 正常来说会有一个[197, 768] -> [197, 3072]再还原为[197, 3072] -> [197, 768]
        # 的步骤，如果不这样处理的话，那就维度一直是768
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()  # GELU激活函数
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        # MLP的结构简洁明了
        return x


class GLU(nn.Module):
    def __init__(self, act_layer):
        super(GLU, self).__init__()
        self.act = act_layer()  # GELU激活函数

    def forward(self, x):
        nc = x.size(2)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc / 2)
        return x[:, :, :nc] * self.act(x[:, :, nc:])


class GluLayer(nn.Module):
    def __init__(self, input_size, output_size, act_layer):
        super().__init__()
        # 第一个线性层
        self.fc1 = nn.Linear(input_size, output_size)
        # 第二个线性层
        self.fc2 = nn.Linear(input_size, output_size)
        # pytorch的GLU层
        self.glu = GLU(act_layer)

    def forward(self, x):
        # 先计算第一个线性层结果
        a = self.fc1(x)
        # 再计算第二个线性层结果
        b = self.fc2(x)
        # 拼接a和b，水平扩展的方式拼接
        # 然后把拼接的结果传给glu
        return self.glu(torch.cat((a, b), dim=2))

class Block(nn.Module):
    def __init__(self,
                 dim=64,
                 num_heads=16,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)  # dim为128
        # self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #                       attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        #  这个地方使用的是GAU结构替代多头机制
        self.attn = GAU(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)  # 这个就是MLP[197, 768] -> [197, 3072]
        # 3072就是768的4倍，正好作为MLP的隐藏层，对于2016数据就是32*4
        # 原始的mlp层
        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)
        # 改进后的FFN_GEGLU
        self.mlp = GluLayer(input_size=dim, output_size=dim, act_layer=act_layer)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    # def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
    #              embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
    #              qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
    #              attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
    #              act_layer=None):
    def __init__(self, img_size=1024, patch_size=(2, 64), in_c=1, num_classes=24,
                 embed_dim=64, depth=4, num_heads=16, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1  # 不知道这个蒸馏是个什么意思,默认为1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        # 这个partial是一个偏函数，就是当函数的参数一直是固定的时候，为了方便
        # 反复的调用，就把这个带有固定参数的函数通过partial赋值为一个值，调用这个值就能够
        # 调用函数
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        # 得到patch_embedding
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        # 这个dist_token是个什么东西啊，原论文中没有这个东西
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        # position embedding也是随机生成的



        self.pos_drop = nn.Dropout(p=drop_ratio)  # 默认的drop_ratio为0，也就是dropout层没用
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        # 随机生成不同的drop概率，drop_path_ratio为0，所以没有用
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
            # 这个for循环为什么不放在前面呢，放在这个地方真的能实现encoder的堆叠嘛
        ])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()
        # 输出最终的num_classes的分类结果

        # Weight init,如果使用正余弦初始化position embedding那么这一段代码会被注释
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)
        # 把位置embedding和cls_token都进行正态分布初始化

    def forward_features(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768] patch embedding
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)  # 得到输入的总embedding
        x = self.blocks(x)  # 输入到encoder中
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def get_features(self, x):
        # 调用 forward_features 方法获取特征
        features = self.forward_features(x)
        if self.dist_token is not None:
            # 如果使用了蒸馏 token，返回两个 token 的特征
            return torch.cat(features, dim=1)
        else:
            # 只返回 cls_token 的特征
            return features.unsqueeze(1).flatten(1)  # 转换为二维数组

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # 这里的headlist没用
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x
    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # 这里的headlist没用
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")  # 之前使用的是kaiming初始化方法
        # nn.init.xavier_normal_(m.weight, gain=1)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
    # 这个部分是对线性层卷积层归一化都进行权重的初始化

def vit_(num_classes: int = 24, has_logits: bool = False):
    """
    ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: converted weights not currently available, too large for github release hosting.
    """
    model = VisionTransformer(img_size=1024,
                              patch_size=(2, 64),
                              embed_dim=256,
                              depth=8,
                              num_heads=16,
                              representation_size=1280 if has_logits else None,
                              num_classes=num_classes)
    return model
