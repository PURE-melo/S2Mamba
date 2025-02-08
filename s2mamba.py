import time
import math
import copy
from functools import partial
from typing import Optional, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"
import numpy as np
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=1, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size), nn.GELU())
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None
    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x

class PCS(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        d_spectral=121,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model/ 16 ) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()
        self.K = 4
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        ##############################################
        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.K, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=self.K, merge=True)
        self.forward_core = self.forward_corev1

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        Bs = Bs.view(B, K, -1, L)
        Cs = Cs.view(B, K, -1, L)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        Ds = self.Ds.view(-1) # (k * d)
        dt_projs_bias = self.dt_projs_bias.view(-1)
        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float32

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0].float() + inv_y[:, 0].float() + wh_y.float() + invwh_y.float()
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y).to(x.dtype)
        return y
    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y = self.forward_core(x)
        y = y*F.gelu(z)
        out = self.out_proj(y)  
        if self.dropout is not None:
            out = self.dropout(out)
        return out
    
class BSS(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        d_spectral=121,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.d_spectral = d_spectral
        self.dt_rank = math.ceil(self.d_model/ 16 ) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()
        ##############################################
        self.x_proj_spectral = (
            nn.Linear(self.d_spectral, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_spectral, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight_spectral = nn.Parameter(torch.stack([t.weight for t in self.x_proj_spectral], dim=0))
        del self.x_proj_spectral
        ##############################################
        self.dt_projs_spectral = (
            self.dt_init(self.dt_rank, self.d_spectral, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_spectral, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight_spectral = nn.Parameter(torch.stack([t.weight for t in self.dt_projs_spectral], dim=0))
        self.dt_projs_bias_spectral = nn.Parameter(torch.stack([t.bias for t in self.dt_projs_spectral], dim=0))
        del self.dt_projs_spectral
        ##############################################
        self.A_logs_spectral = self.A_log_init(self.d_state, self.d_spectral, copies=2, merge=True)
        self.Ds_spectral = self.D_init(self.d_spectral, copies=2, merge=True)
        self.out_norm_spectral = nn.LayerNorm(self.d_spectral)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D


    def forward_spectral(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1
        old_B, old_C, old_H, old_W = x.shape
        x = torch.transpose(x.contiguous().view(old_B, old_C, -1), 1, 2).view(old_B, old_H*old_W, old_C, 1)
        B, C, H, W = x.shape
        L = H * W
        K = 2
        xs = torch.cat([x, torch.flip(x, dims=[-1])], dim=1)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight_spectral)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight_spectral)
        xs = xs.view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        Bs = Bs.view(B, K, -1, L)
        Cs = Cs.view(B, K, -1, L)        
        As = -torch.exp(self.A_logs_spectral.float()).view(-1, self.d_state)
        Ds = self.Ds_spectral.view(-1)
        dt_projs_bias = self.dt_projs_bias_spectral.view(-1)
        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float32
        inv_y = torch.flip(out_y, dims=[-1])
        y = out_y[:, 0].float() + inv_y[:, 1].float()
        
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm_spectral(y).to(x.dtype)
        
        y = torch.transpose(y.view(B, L, -1), 1, 2).view(old_B, old_H, old_W, old_C)

        return y
    
    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y_spectral = self.forward_spectral(x)
        y = y_spectral*F.gelu(z)
        out = self.out_proj(y) 
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class MambaBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        d_spectral: int = 121,
        **kwargs,
    ):
        super().__init__()
        self.ln = norm_layer(hidden_dim)
        self.PCS = PCS(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, d_spectral=d_spectral, **kwargs)
        self.BSS = BSS(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, d_spectral=d_spectral, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        input_ln = self.ln(input)
        x_spatial, x_spectral = self.drop_path(self.PCS(input_ln)), self.drop_path(self.BSS(input_ln))
        return x_spatial, x_spectral
    


class FeedForward_softmax(nn.Module):
    def __init__(self, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        return self.net(x)
class SMG(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        d_spectral: int = 121,
        **kwargs,
    ):
        super().__init__()
        self.FFN_softmax = FeedForward_softmax(hidden_dim*2, attn_drop_rate)
        self.threshold = 0.1
        self.drop_path = DropPath(drop_path)

    def forward(self, input_spatial: torch.Tensor, input_spectral: torch.Tensor):
        gate = self.FFN_softmax(torch.cat([input_spatial, input_spectral], dim=-1))
        gate1, gate2 = gate[:,:,:,0].unsqueeze(-1), gate[:,:,:,1].unsqueeze(-1)
        gate1, gate2 = torch.where(gate1<self.threshold, torch.zeros_like(gate1), gate1), torch.where(gate2<self.threshold, torch.zeros_like(gate2), gate2)
        x = gate1*input_spatial + gate2*input_spectral
        return self.drop_path(x)

class MambaLayer(nn.Module):
    def __init__(
        self, 
        dim, 
        depth, 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        downsample=None, 
        d_state=16,
        d_spectral=121,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim

        self.blocks = nn.ModuleList([
            MambaBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
                d_spectral=d_spectral,
            )
            for i in range(depth)])
        
        self.fusion = SMG(
                hidden_dim=dim,
                drop_path=attn_drop,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
                d_spectral=d_spectral,
            )
    def forward(self, x):
        for blk in self.blocks:
            x_spatial, x_spectral = blk(x)
            x = x+self.fusion(x_spatial, x_spectral)
        return x
    
class S2Mamba(nn.Module):
    def __init__(self, patch=11, in_chans=154, num_classes=16, depths=[1], dims=[64], d_state=16, drop_rate=0., attn_drop_rate=0.0, drop_path_rate=0.1, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims
        self.patch_embed = PatchEmbed(patch_size=1, in_chans=in_chans, embed_dim=self.embed_dim, norm_layer=None)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = MambaLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=None, 
                d_spectral = patch**2,
            )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        #elif isinstance(m, nn.LayerNorm):
            #nn.init.constant_(m.bias, 0)
            #nn.init.constant_(m.weight, 1)
    def forward(self, x):
        x = self.patch_embed(x) 
        for layer in self.layers:
            x = layer(x)
        x = torch.flatten(x, 1, 2)
        feats = self.norm(x)
        x = self.avgpool(feats.transpose(1, 2)) 
        agg_feats = torch.flatten(x, 1)
        x = self.head(agg_feats)
        return x
