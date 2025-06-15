import torch
import torch.nn as nn
from einops import rearrange
from bisect import bisect_left
import numpy as np

from timm.models.layers import DropPath, Mlp
from taming.modules.diffusionmodules.tome import Block
import taming.modules.diffusionmodules.tome as tome


class ResBlock(nn.Module):
    def __init__(self, 
                 in_filters,
                 out_filters,
                 use_conv_shortcut = False,
                 use_agn = False,
                 ) -> None:
        super().__init__()

        self.in_filters = in_filters
        self.out_filters = out_filters
        self.use_conv_shortcut = use_conv_shortcut
        self.use_agn = use_agn

        if not use_agn: ## agn is GroupNorm likewise skip it if has agn before
            self.norm1 = nn.GroupNorm(32, in_filters, eps=1e-6)
        self.norm2 = nn.GroupNorm(32, out_filters, eps=1e-6)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(3, 3), padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_filters, out_filters, kernel_size=(3, 3), padding=1, bias=False)

        if in_filters != out_filters:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_filters, out_filters, kernel_size=(3, 3), padding=1, bias=False)
            else:
                self.nin_shortcut = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), padding=0, bias=False)

    def forward(self, x, **kwargs):
        residual = x

        if not self.use_agn:
            x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)
        if self.in_filters != self.out_filters:
            if self.use_conv_shortcut:
                residual = self.conv_shortcut(residual)
            else:
                residual = self.nin_shortcut(residual)

        return x + residual


class MixMlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            drop=0.,
            **kwargs):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, (3, 3), 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        B, N, C = x.shape
        sz = int(N ** 0.5)
        x = x.transpose(1, 2).view(B, C, sz, sz)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, *, embed_dim=768, num_layers=12, num_heads=12, drop_path=0.0,
                 with_cls_token=True, with_dw_conv=False, init_values=1e-5,
                 use_flash_attn=False, **kwargs):
        super(TransformerBlock, self).__init__()

        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.with_cls_token = with_cls_token
        self.with_dw_conv = with_dw_conv

        if self.with_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        if self.with_dw_conv:
            self.pos_embed = nn.Conv2d(
                self.embed_dim, self.embed_dim, (3, 3), padding=1, groups=self.embed_dim)
        self.pos_drop = nn.Dropout(p=0.0)

        # dp_rates=[x.item() for x in torch.linspace(0, drop_path, num_layers)]
        dp_rates=[x.item() for x in torch.linspace(drop_path, 0.0, num_layers)]
        self.blocks = nn.Sequential(
            *[Block(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                init_values=init_values,
                drop_path=dp_rates[j],
                norm_layer=nn.LayerNorm,
                act_layer=nn.GELU,
                mlp_layer=Mlp if not with_dw_conv else MixMlp,
                use_flash_attn=use_flash_attn,
            ) for j in range(num_layers)]
        )
        # self.norm = nn.LayerNorm(self.embed_dim)

        self.init_weights()

    def init_weights(self):
        if self.with_cls_token:
            nn.init.normal_(self.cls_token, std=1e-6)
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                torch.nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self,x):
        B, N, C = x.shape
        if self.with_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        if self.with_dw_conv:
            sz = int(N ** 0.5)
            x = x.transpose(1, 2).view(B, C, sz, sz)
            x = x + self.pos_embed(x)
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        x = self.blocks(x)
        # x = self.norm(x)
        return x


class MergeVQEncoder(nn.Module):
    def __init__(self, *, ch, out_ch, in_channels, ch_mult=(1,2,4,8), resolution=256, z_channels=18,
                 num_res_blocks=2, num_att_blocks=12, num_heads=8, head_dim=None, drop_path=0,
                 merge_ratio=None, merge_num=None, dist_head=None, num_classes=None, model_path=None,
                 num_rep_blocks=0, cand_distribution=None, isotropic=False, use_flash_attn=False,
                 **ignore_kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.z_channels = z_channels
        self.resolution = resolution
        self.dist_head = dist_head
        self.isotropic = isotropic

        self.num_blocks = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.num_att_blocks = num_att_blocks
        if isinstance(self.num_res_blocks, list):
            assert len(self.num_res_blocks) == self.num_blocks
        elif isinstance(self.num_res_blocks, int):
            self.num_res_blocks = [num_res_blocks for i in range(self.num_blocks)]

        ## downsampling
        if not self.isotropic:
            self.conv_in = nn.Conv2d(in_channels, ch,
                                     kernel_size=(3, 3), padding=1, bias=False)

        ## construct the model
        self.down = nn.ModuleList()
        curr_res = resolution
        # classical CNN
        in_ch_mult = (1,) + tuple(ch_mult)
        for i_level in range(self.num_blocks):
            block = nn.ModuleList()
            block_dim = int(ch * in_ch_mult[i_level])  # [1, 1, 2, 2, 4]
            block_out = int(ch * ch_mult[i_level])  # [1, 2, 2, 4]
            for j in range(self.num_res_blocks[i_level]):
                block.append(ResBlock(block_dim, block_out))
                block_dim = block_out

            down = nn.Module()
            if not self.isotropic:  # build CNN encoder
                down.block = block
            # post-downsample for classical CNN
            if i_level < self.num_blocks - 1:
                down.downsample = nn.Conv2d(block_out, block_out, kernel_size=(3, 3), stride=(2, 2), padding=1)
                curr_res = curr_res // 2
            if not self.isotropic:  # build CNN downsampling
                self.down.append(down)
        # pure ViT encoder
        if self.isotropic:
            patch_size = resolution // curr_res
            block_dim = int(self.isotropic) if isinstance(self.isotropic, int) and self.isotropic > 1 else block_dim
            self.conv_in = nn.Sequential(
                nn.Conv2d(in_channels, block_dim, kernel_size=(patch_size, patch_size),
                    stride=patch_size, padding=0),
                nn.GroupNorm(block_dim, block_dim, eps=1e-6),
            )

        ## middle
        self.embed_dim = block_dim
        self.num_heads = num_heads if head_dim is None else max(self.embed_dim // head_dim, num_heads)
        self.merge_num = merge_num
        self.embed_res = curr_res
        self.embed_len, self.merge_res = None, None
        assert self.embed_dim % self.num_heads == 0, 'dim should be divisible by num_heads'
        self.attn = TransformerBlock(embed_dim=self.embed_dim, num_layers=self.num_att_blocks,
                                     num_heads=self.num_heads, with_cls_token=True, drop_path=drop_path,
                                     use_flash_attn=use_flash_attn)
        self.attn = self.apply_merge(self.attn, merge_ratio=(merge_ratio, -0.5),
                                     merge_num=merge_num, cand_distribution=cand_distribution)

        ## generation output
        self.norm_out = nn.GroupNorm(32, self.embed_dim, eps=1e-6)
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv2d(self.embed_dim, z_channels, kernel_size=(1, 1))

        ## representation output
        if self.dist_head:
            hidden_dim = int(4 * self.embed_dim)
            num_classes = self.embed_dim if not num_classes else num_classes
            self.num_rep_blocks = num_rep_blocks
            self.rep_attn = TransformerBlock(embed_dim=self.embed_dim, num_layers=num_rep_blocks,
                                             num_heads=self.num_heads, with_cls_token=False,
                                             use_flash_attn=use_flash_attn) if num_rep_blocks > 0 else None
            if self.dist_head == 'layer3':
                self.dist_head = nn.Sequential(
                    nn.Linear(self.embed_dim, hidden_dim, bias=True),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim, bias=True),
                    nn.GELU(),
                    nn.Linear(hidden_dim, num_classes, bias=True),
                )
            else:
                self.dist_head = nn.Sequential(
                    nn.Linear(self.embed_dim, hidden_dim, bias=True),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, num_classes, bias=True),
                )
        if model_path is not None:
            self.load_pretrained_vit(model_path)

    def forward(self, x, return_head=False):
        ## downsampling
        x = self.conv_in(x)
        if not self.isotropic:  # CNN encoder
            for i_level in range(self.num_blocks):
                for i_block in range(self.num_res_blocks[i_level]):
                    x = self.down[i_level].block[i_block](x)
                if i_level <  self.num_blocks - 1:
                    x = self.down[i_level].downsample(x)

        ## middle
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)
        x = self.attn(x)

        ## rep output
        if self.dist_head and return_head:
            if self.rep_attn is not None:
                x = self.rep_attn(x)

            cls_token = x[:, 0, :]
            cls_token = self.dist_head(cls_token)
            return cls_token

        ## gen output
        x = x[:, 1:, :].permute(0, 2, 1).reshape(B, C, self.merge_res, self.merge_res)
        x = self.norm_out(x)
        x = self.act_out(x)
        x = self.conv_out(x)
        return x

    def apply_merge(self, model, merge_ratio=None, merge_num=None, cand_distribution="gaussian-5"):
        # apply ToMe
        tome.timm(model)
        
        # check ToMe settings
        inflect, num_layers = -0.5, self.num_att_blocks
        if isinstance(merge_ratio, tuple):
            merge_ratio, inflect = merge_ratio
        if merge_ratio is not None and merge_num is None:
            self.merge_num = sum(tome.parse_r(num_layers, r=(merge_ratio, inflect)))
        elif merge_ratio is None and merge_num is not None or cand_distribution is not None:
            merge_ratio = tome.check_parse_r(num_layers, merge_num, self.embed_res ** 2, inflect)
            if isinstance(cand_distribution, str):
                cand_distribution = cand_distribution.split('-')
                if len(cand_distribution) == 2:
                    cand_distribution, r_cand_num = cand_distribution[0], int(cand_distribution[-1])
                else:
                    cand_distribution, r_cand_num = cand_distribution[0], 5
            if cand_distribution is not None:
                # generate candidate list with the center index
                bias = 1 if cand_distribution.lower() == "gaussian" else 0
                remain_list = [int(((self.embed_res**2 - merge_num) **0.5 + i) **2) \
                               for i in range(int(-bias), r_cand_num - bias)]
                merged_list = [self.embed_res**2 - num for num in remain_list]
                r_prob_list = generate_prob_list(remain_list, distribution=cand_distribution,
                                                 mean=bias, std=0.5, lambd=2.0)
                r_cand_list = [tome.check_parse_r(num_layers, int(self.embed_res**2 - num), self.embed_res**2, inflect
                                                  ) for num in remain_list]
                model._tome_info["remain_list"] = remain_list
                model._tome_info["r_cand_list"] = r_cand_list
                model._tome_info["r_prob_list"] = r_prob_list
        else:
            self.merge_num, self.merge_ratio = merge_num, merge_ratio
        self.embed_len = int(self.embed_res ** 2)
        self.merge_res = int((self.embed_len - self.merge_num) ** 0.5)

        # update
        model.r = (merge_ratio, inflect)
        model._tome_info["r"] = model.r
        model._tome_info["total_merge"] = merge_num
        return model

    def sampling_r_candidate(self, rand=None):
        if self.attn._tome_info.get("r_cand_list", None) is None:
            return rand
        if rand is None:
            rand = np.random.random()
        # get the index of the first cumulative probability
        index = bisect_left(self.attn._tome_info["r_prob_list"], rand)
        r = self.attn._tome_info["r_cand_list"][index]
        if self.attn.r[0] != r:
            self.attn.r = (r, -0.5)
            self.attn._tome_info["total_merge"] = int(self.embed_len - self.attn._tome_info["remain_list"][index])
            self.merge_res = int(self.attn._tome_info["remain_list"][index] ** 0.5)
        return (r, -0.5)

    def load_pretrained_vit(self, model_path=None):
        """loading pre-trained model with weight selection"""
        from collections import OrderedDict
        from timm import create_model

        state_dict = OrderedDict()
        if 'pth' in model_path:
            teacher_weights = torch.load(model_path)
        else:
            teacher_weights = create_model(model_path, pretrained=True).state_dict()
        if 'model' in teacher_weights.keys():
            teacher_weights = teacher_weights['model']
        # weight selection
        student_weights = self.attn.state_dict()
        weight_selection = {}
        for key in student_weights.keys():
            if ('block' in key or 'cls_token' in key) and key in teacher_weights.keys():
                weight_selection[key] = uniform_element_selection(teacher_weights[key], student_weights[key].shape)
        # load to attention
        print("load pre-trained model for encoder:\n",
              self.attn.load_state_dict(weight_selection, strict=False))


def uniform_element_selection(tea_weights, stu_shape):
    """Large Model Initialization (https://arxiv.org/abs/2311.18823)"""
    assert tea_weights.dim() == len(stu_shape), "Tensors have different number of dimensions"
    tea_weights = tea_weights.clone()
    if tea_weights.shape != stu_shape:
        for dim in range(tea_weights.dim()):
            assert tea_weights.shape[dim] >= stu_shape[dim], "Teacher's dimension should not be smaller than students'"
            if tea_weights.shape[dim] % stu_shape[dim] == 0:
                step = tea_weights.shape[dim] // stu_shape[dim]
                indices = torch.arange(stu_shape[dim]) * step
            else:
                indices = torch.round(torch.linspace(0, tea_weights.shape[dim]-1, stu_shape[dim])).long()
            tea_weights = torch.index_select(tea_weights, dim, indices)
    else:
        assert tea_weights.shape == stu_shape, "Selected weight should be the same as student"
    return tea_weights


def generate_prob_list(r_cand_list, distribution='gaussian', **kwargs):
    """Generate r_prob_list (Cumulative density)

    Parameters:
        r_cand_list (list): List of candidate values
        distribution (str): Type of probability distribution
        **kwargs: Additional parameters for the distribution
    """
    n = len(r_cand_list)
    r_cand_list = np.arange(n)
    r_prob = np.zeros(n)

    if distribution.lower() == 'gaussian':
        mean = kwargs.get('mean', 0)  # Default mean is 0
        std = kwargs.get('std', 1)    # Default standard deviation is 1
        r_prob = np.exp(-((np.array(r_cand_list) - mean) ** 2) / (2 * std ** 2))
        r_prob += 0.05
    elif distribution.lower() == 'exponential':
        lambd = kwargs.get('lambd', 1)  # Default lambda is 1
        r_prob = lambd * np.exp(-lambd * np.array(r_cand_list))    
        r_prob += 0.1
    else:
        r_prob = np.ones(n) / n

    r_prob /= np.sum(r_prob)
    r_prob_list = np.around(np.cumsum(r_prob), decimals=5)
    # print(r_prob_list)

    return r_prob_list


class MergeVQDecoder(nn.Module):
    def __init__(self, *, ch, out_ch, in_channels, ch_mult=(1, 2, 2, 4), resolution=256, z_channels=18,
                 num_res_blocks=2, num_att_blocks=12, num_res_extra=1, num_heads=8, head_dim=None,
                 drop_path=0, merge_ratio=None, use_flash_attn=False,
                 **ignore_kwargs):
        super().__init__()

        self.ch = ch
        self.num_blocks = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.num_att_blocks = num_att_blocks
        self.num_res_extra = num_res_extra
        self.resolution = resolution
        self.in_channels = in_channels
        if isinstance(num_res_blocks, list):
            assert len(self.num_res_blocks) == self.num_blocks
        elif isinstance(num_res_blocks, int):  # extra one block for Decoder
            self.num_res_blocks = [num_res_blocks + num_res_extra for i in range(self.num_blocks)]
            self.num_res_blocks[0] = self.num_res_blocks[0] - self.num_res_extra

        block_in = int(ch * ch_mult[self.num_blocks - 1])

        self.conv_in = nn.Conv2d(
            z_channels, block_in, kernel_size=(3, 3), padding=1, bias=True)

        # middle
        self.embed_dim = block_in
        self.num_heads = num_heads
        self.num_heads = num_heads if head_dim is None else max(self.embed_dim // head_dim, num_heads)
        assert self.embed_dim % self.num_heads == 0, 'dim should be divisible by num_heads'
        self.attn = TransformerBlock(
            embed_dim=self.embed_dim, num_layers=self.num_att_blocks, num_heads=self.num_heads,
            with_cls_token=False, with_dw_conv=True, drop_path=0.0, use_flash_attn=use_flash_attn)

        # upsampling
        self.up = nn.ModuleList()
        self.adaptive = nn.ModuleList()

        for i_level in reversed(range(self.num_blocks)):
            block = nn.ModuleList()
            block_out = int(ch * ch_mult[i_level])
            self.adaptive.insert(0, AdaptiveGroupNorm(z_channels, block_in))
            for i_block in range(self.num_res_blocks[i_level]):
                block.append(ResBlock(block_in, block_out))
                block_in = block_out
            
            up = nn.Module()
            up.block = block
            if i_level > 0:
                up.upsample = Upsampler(block_in)
            self.up.insert(0, up)

        self.norm_out = nn.GroupNorm(32, block_in, eps=1e-6)
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=(3, 3), padding=1)

    def forward(self, z):
        """decoding with AdaGN"""
        style = z.clone()  # for adaptive groupnorm
        ## in
        z = self.conv_in(z)
        # print('conv in', z.shape)

        # middle
        B, C, H, W = z.shape
        z = z.view(B, C, -1).permute(0, 2, 1)
        # print('attn in', z.shape)
        z = self.attn(z)
        z = z.permute(0, 2, 1).reshape(B, C, H, W)

        # upsampling
        for i_level in reversed(range(self.num_blocks)):
            ### pass in each resblock first adaGN
            z = self.adaptive[i_level](z, style)
            for i_block in range(self.num_res_blocks[i_level]):
                z = self.up[i_level].block[i_block](z)
            
            if i_level > 0:
                z = self.up[i_level].upsample(z)

        z = self.norm_out(z)
        z = self.act_out(z)
        z = self.conv_out(z)

        return z


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int = 512, num_heads: int = 8, mlp_ratio: float = 4.,
                 qkv_bias: bool = False, attn_drop: float = 0., proj_drop: float = 0.,
                 drop_path: float = 0., init_values: float = 1e-5,
                 act_layer: nn.Module = nn.GELU, norm_layer: nn.Module = nn.LayerNorm,
                 mlp_layer: nn.Module = Mlp, **ignore_kwargs) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.norm1 = norm_layer(dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.gamma1 = nn.Parameter(init_values * torch.ones(dim))

        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.norm2 = norm_layer(dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.gamma2 = nn.Parameter(init_values * torch.ones(dim))

    def forward_attention(self, q, kv):
        B, N, C = q.shape
        q = self.q(q).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        B, L, C = kv.shape
        kv = self.kv(kv).reshape(B, L, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward(self, q, kv):
        q = q + self.drop_path1(self.gamma1 * self.forward_attention(self.norm1(q), kv))
        q = q + self.drop_path2(self.gamma2 * self.mlp(self.norm2(q)))
        return q


class SourceRecovery(nn.Module):
    """ Recovery Source Matrix and Full Tokens """
    def __init__(self, *, keep_num=None, full_num=256, embed_dim=18, rec_embed_dim=384,
                 num_layers=2, num_heads=8, learn_source=None, **ignore_kwargs):
        super().__init__()
        self.keep_num = keep_num
        self.full_num = full_num
        self.embed_dim = embed_dim
        self.rec_embed_dim = rec_embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.learn_source = learn_source

        if self.learn_source:
            self.rec_embed = nn.Linear(embed_dim, self.rec_embed_dim, bias=True)
            self.rec_tokens = nn.Parameter(
                torch.zeros(1, self.full_num, self.rec_embed_dim), requires_grad=True)
            self.norm_in = nn.LayerNorm(self.rec_embed_dim)

            self.blocks = nn.Sequential(
                *[CrossAttentionBlock(
                    dim=self.rec_embed_dim,
                    num_heads=self.num_heads,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    drop_path=0.1,
                    norm_layer=nn.LayerNorm,
                    act_layer=nn.GELU,
                    mlp_layer=Mlp,
                ) for _ in range(self.num_layers)]
            )
            self.rec_head = nn.Linear(self.rec_embed_dim, self.rec_embed_dim, bias=True)

            self.init_weights()

    def init_weights(self):
        if self.rec_tokens is not None:
            nn.init.normal_(self.rec_tokens, std=1e-6)
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                torch.nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
                nn.init.constant_(module.weight, val=1.)
                nn.init.constant_(module.bias, val=0.)

    def forward(self, x, source=None):
        batchsize, seq_len, _ = x.shape
        if self.learn_source:
            kv = self.rec_embed(x)
            kv = self.norm_in(kv)

            # cross attention
            for i in range(self.num_layers):
                if i == 0:
                    q = self.rec_tokens
                q = self.blocks[i](q, kv)
                # print(i, q.shape)

            # prediction
            q = self.rec_head(q)
            k = self.rec_head(kv)
            source = q @ k.transpose(-2, -1)
            source = source.softmax(dim=-1)  # [B, full_L, keep_L]
            # print('source', source.shape)

            # indexing
            with torch.no_grad():
                source_index = torch.argmax(source, dim=-1)
                source_index = nn.functional.one_hot(source_index, num_classes=k.shape[1]).transpose(-2, -1)
                x = token_unmerge(x, source_index)
                # print('unmerge', x.shape, source_index.shape)

            return (x, source.transpose(-2, -1))  # [B, keep_L, full_L]
        else:
            x = token_unmerge(x, source)
            return x


def token_unmerge(keep_tokens, source=None):
    """ recovery full tokens with the given source_matrix """
    if source is None:
        return keep_tokens
    B, _, C = keep_tokens.shape
    full_L = source.shape[-1]  # [B, keep_L, full_L]
    full_tokens = torch.zeros(B, full_L, C).to(keep_tokens)
    indices = (source == 1).nonzero(as_tuple=False)

    batch_idx = indices[:, 0]
    full_tokens[batch_idx, indices[:, 2], :] = keep_tokens[batch_idx, indices[:, 1], :]

    return full_tokens


def depth_to_space(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """ Depth-to-Space DCR mode (depth-column-row) core implementation.

        Args:
            x (torch.Tensor): input tensor. The channels-first (*CHW) layout is supported.
            block_size (int): block side size
    """
    # check inputs
    if x.dim() < 3:
        raise ValueError(
            f"Expecting a channels-first (*CHW) tensor of at least 3 dimensions"
        )
    c, h, w = x.shape[-3:]

    s = block_size**2
    if c % s != 0:
        raise ValueError(
            f"Expecting a channels-first (*CHW) tensor with C divisible by {s}, but got C={c} channels"
        )

    outer_dims = x.shape[:-3]

    # splitting two additional dimensions from the channel dimension
    x = x.view(-1, block_size, block_size, c // s, h, w)

    # putting the two new dimensions along H and W
    x = x.permute(0, 3, 4, 1, 5, 2)

    # merging the two new dimensions with H and W
    x = x.contiguous().view(*outer_dims, c // s, h * block_size,
                            w * block_size)

    return x


class Upsampler(nn.Module):
    def __init__(self, dim, dim_out = None):
        super().__init__()
        dim_out = dim * 4 if dim_out is None else dim_out
        self.conv1 = nn.Conv2d(dim, dim_out, (3, 3), padding=1)
        self.depth2space = depth_to_space

    def forward(self, x):
        """ input_image: [B C H W] """
        out = self.conv1(x)
        out = self.depth2space(out, block_size=2)
        return out


class AdaptiveGroupNorm(nn.Module):
    def __init__(self, z_channel, in_filters, num_groups=32, eps=1e-6):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups=32, num_channels=in_filters, eps=eps, affine=False)
        self.gamma = nn.Linear(z_channel, in_filters)
        self.beta = nn.Linear(z_channel, in_filters)
        self.eps = eps

    def forward(self, x, quantizer):
        B, C, _, _ = x.shape
        ### calcuate var for scale
        scale = rearrange(quantizer, "b c h w -> b c (h w)")
        scale = scale.var(dim=-1) + self.eps #not unbias
        scale = scale.sqrt()
        scale = self.gamma(scale).view(B, C, 1, 1)

        ### calculate mean for bias
        bias = rearrange(quantizer, "b c h w -> b c (h w)")
        bias = bias.mean(dim=-1)
        bias = self.beta(bias).view(B, C, 1, 1)

        x = self.gn(x)
        x = scale * x + bias

        return x


if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    resolution = 256
    cand_distribution = 'gaussian-6'
    cand_sample_times = 0

    # ch, ch_mult = 64, (1, 2, 4, 8)
    # num_att_blocks, num_res_blocks, r, merge_num = 12, 4, None, 768
    ch, ch_mult = 64, (1, 1, 2, 4, 8)
    num_att_blocks, num_res_blocks, r, merge_num = 12, 4, None, 112
    # ch, ch_mult = 64, (1, 2, 4, 8, 16)
    # num_att_blocks, num_res_blocks, r, merge_num = 12, 4, None, 112
    num_heads = 8

    isotropic = 768
    model_path = None
    # model_path = 'vit_base_patch14_dinov2.lvd142m'  # cp -r .cache/clip-vit_model/models--timm--vit_base_patch14_dinov2.lvd142m ~/.cache/huggingface/hub/
    # model_path = 'vit_base_patch16_clip_224.laion2b'  # cp -r .cache/clip-vit_model/models--laion--CLIP-ViT-B-16-laion2B-s34B-b88K ~/.cache/huggingface/hub/
    # model_path = 'vit_base_patch16_siglip_224.webli'  # cp -r .cache/clip-vit_model/models--timm--ViT-B-16-SigLIP ~/.cache/huggingface/hub/
    # model_path = 'vit_large_patch14_clip_224.laion2b'  # cp -r .cache/clip-vit_model/models--laion--CLIP-ViT-L-14-laion2B-s32B-b82K ~/.cache/huggingface/hub/

    model = MergeVQEncoder(
        ch=ch, out_ch=3, in_channels=3, ch_mult=ch_mult, resolution=resolution, z_channels=18,
        num_res_blocks=num_res_blocks, num_att_blocks=num_att_blocks, 
        num_heads=num_heads, head_dim=None, merge_ratio=r, merge_num=merge_num,
        dist_head=None, num_classes=None, num_rep_blocks=0, model_path=model_path,
        cand_distribution=cand_distribution, isotropic=isotropic, use_flash_attn=False)

    # import pdb; pdb.set_trace()
    if cand_distribution and cand_sample_times > 0:
        print('r_cand_list:', model.attn._tome_info['r_cand_list'])
        print('remain_list:', model.attn._tome_info['remain_list'])
        print('r_prob_list:', model.attn._tome_info['r_prob_list'])
        for i in range(cand_sample_times):
            model.sampling_r_candidate()
            print("out:", i, model.attn.r, model.merge_res ** 2)
            x = torch.randn(1, 3, resolution, resolution)

    x = torch.randn(1, 3, resolution, resolution)
    flop = FlopCountAnalysis(model, x)
    y = model(x, return_head=False)
    source = model.attn._tome_info['source']
    print('encoder (r={}): {}'.format(model.attn.r, y.shape))
    print(flop_count_table(flop, max_depth=4))
    print('MACs (G) of Encoder: {:.3f}'.format(flop.total() / 1e9))

    if source is not None:
        print('encoder source matrix:', source.shape)

        y = y.reshape(1, 18, -1)
        y = y.permute(0, 2, 1)
        recovery = SourceRecovery(
            keep_num=y.shape[-1], full_num=source.shape[-1],
            embed_dim=18, rec_embed_dim=384, num_layers=4, num_heads=8, learn_source=True)
        output = recovery(y, source[:, 1:, 1:])
        if isinstance(output, tuple):
            print('source_recovery x=', output[0].shape, 'source=', output[1].shape)
        else:
            print('source_recovery x=', output.shape)

    model = MergeVQDecoder(
        ch=ch, out_ch=3, in_channels=3, ch_mult=ch_mult, resolution=resolution, z_channels=18,
        num_res_blocks=num_res_blocks, num_att_blocks=num_att_blocks, num_res_extra=1,
        num_heads=num_heads)
    curr_res = resolution // 2 ** (len(ch_mult) - 1)
    x = torch.randn(1, 18, curr_res, curr_res)
    flop = FlopCountAnalysis(model, x)
    y = model(x)
    print(flop_count_table(flop, max_depth=4))
    print('MACs (G) of Decoder: {:.3f}'.format(flop.total() / 1e9))
    # print(model)
