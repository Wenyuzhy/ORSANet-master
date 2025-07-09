import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from typing import Any, Optional, Tuple, Type
from torchvision.models import convnext_large, convnext_base, convnext_small, convnext_tiny, swin_b, swin_v2_b, swin_v2_s, swin_v2_t, mobilenet_v3_large, efficientnet_v2_m
import pdb
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from .transformer import *
from .utils_models import *

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
    
class FaceDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: 256,
        transformer: nn.Module,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.background_token = nn.Embedding(1, transformer_dim)
        self.face_token = nn.Embedding(1, transformer_dim)
        self.leftbro_token = nn.Embedding(1, transformer_dim)
        self.rightbro_token = nn.Embedding(1, transformer_dim)
        self.lefteye_token = nn.Embedding(1, transformer_dim)
        self.righteye_token = nn.Embedding(1, transformer_dim)
        self.nose_token = nn.Embedding(1, transformer_dim)
        self.upperlip_token = nn.Embedding(1, transformer_dim)
        self.innermouth_token = nn.Embedding(1, transformer_dim)
        self.lowerlip_token = nn.Embedding(1, transformer_dim)
        self.hair_token = nn.Embedding(1, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        
        self.output_hypernetwork_mlps = MLP(
            transformer_dim, transformer_dim, transformer_dim // 8, 3
            )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        image_embeddings - torch.Size([1, 256, 128, 128])
        image_pe - torch.Size([1, 256, 128, 128])
        '''
        output_tokens = torch.cat([self.background_token.weight, self.face_token.weight, self.leftbro_token.weight, self.rightbro_token.weight, self.lefteye_token.weight, \
         self.righteye_token.weight, self.nose_token.weight, self.upperlip_token.weight, self.innermouth_token.weight,self.lowerlip_token.weight, self.hair_token.weight], dim=0) 

        tokens = output_tokens.unsqueeze(0).expand(image_embeddings.size(0), -1, -1) ##### torch.Size([4, 11, 256])

        src = image_embeddings ##### torch.Size([4, 256, 128, 128])
        pos_src = image_pe.expand(image_embeddings.size(0), -1, -1, -1)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)  ####### hs - torch.Size([BS, 11, 256]), src - torch.Size([BS, 16348, 256])
        mask_token_out =  hs[:, :, :]
             
        src = src.transpose(1, 2).view(b, c, h, w) ##### torch.Size([4, 256, 128, 128])
        upscaled_embedding = self.output_upscaling(src)  ##### torch.Size([4, 32, 512, 512]) 
        hyper_in = self.output_hypernetwork_mlps(mask_token_out) ##### torch.Size([1, 11, 32])
        b, c, h, w = upscaled_embedding.shape
        seg_output = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)  ##### torch.Size([1, 11, 512, 512])
        
        return seg_output



class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C


class SegfaceMLP(nn.Module):
    """
    Linear Embedding.
    """

    def __init__(self, input_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, 256)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.proj(hidden_states)
        return hidden_states

class SegFaceLapa_infer(nn.Module):
    def __init__(self, input_resolution, model):
        super(SegFaceLapa_infer, self).__init__()
        self.input_resolution = input_resolution
        self.model = model

        if self.model == "swin_base":
            swin_v2 = swin_b(weights='IMAGENET1K_V1')
            self.backbone = torch.nn.Sequential(*(list(swin_v2.children())[:-1]))
            self.target_layer_names = ['0.1', '0.3', '0.5', '0.7']
            self.multi_scale_features = []
        
        if self.model == "swinv2_base":
            swin_v2 = swin_v2_b(weights='IMAGENET1K_V1')
            self.backbone = torch.nn.Sequential(*(list(swin_v2.children())[:-1]))
            self.target_layer_names = ['0.1', '0.3', '0.5', '0.7']
            self.multi_scale_features = []

        if self.model == "swinv2_small":
            swin_v2 = swin_v2_s(weights='IMAGENET1K_V1')
            self.backbone = torch.nn.Sequential(*(list(swin_v2.children())[:-1]))
            self.target_layer_names = ['0.1', '0.3', '0.5', '0.7']
            self.multi_scale_features = []

        if self.model == "swinv2_tiny":
            swin_v2 = swin_v2_t(weights='IMAGENET1K_V1')
            self.backbone = torch.nn.Sequential(*(list(swin_v2.children())[:-1]))
            self.target_layer_names = ['0.1', '0.3', '0.5', '0.7']
            self.multi_scale_features = []

        if self.model == "convnext_base":
            convnext = convnext_base(pretrained=True)
            self.backbone = torch.nn.Sequential(*(list(convnext.children())[:-1]))
            self.target_layer_names = ['0.1', '0.3', '0.5', '0.7']
            self.multi_scale_features = []

        if self.model == "convnext_small":
            convnext = convnext_small(pretrained=True)
            self.backbone = torch.nn.Sequential(*(list(convnext.children())[:-1]))
            self.target_layer_names = ['0.1', '0.3', '0.5', '0.7']
            self.multi_scale_features = []

        if self.model == "convnext_tiny":
            convnext = convnext_small(pretrained=True)
            self.backbone = torch.nn.Sequential(*(list(convnext.children())[:-1]))
            self.target_layer_names = ['0.1', '0.3', '0.5', '0.7']
            self.multi_scale_features = []           
        
        if self.model == "resnet":
            resnet101 = models.resnet101(pretrained=True)
            self.backbone = torch.nn.Sequential(*(list(resnet101.children())[:-1]))
            self.target_layer_names = ['4', '5', '6', '7']
            self.multi_scale_features = []

        if self.model == "mobilenet":
            mobilenet = mobilenet_v3_large(pretrained=True).features
            self.backbone = mobilenet
            self.target_layer_names = ['3', '6', '12', '16']
            self.multi_scale_features = []
        
        if self.model == "efficientnet":
            efficientnet = efficientnet_v2_m(pretrained=True).features
            self.backbone = efficientnet
            self.target_layer_names = ['2', '3', '5', '8']
            self.multi_scale_features = []

        embed_dim = 1024
        out_chans = 256
        
        self.pe_layer = PositionEmbeddingRandom(out_chans // 2)   
        self.get_matrix_fn = functools.partial(get_face_align_matrix, target_shape=(self.input_resolution, self.input_resolution), target_face_scale=1.0)
        self.warp_fn = functools.partial(make_tanh_warp_grid, warp_factor=0.8, warped_shape=(self.input_resolution, self.input_resolution))
        self.inv_warp_fn = functools.partial(make_inverted_tanh_warp_grid, warp_factor=0.8, warped_shape=(self.input_resolution, self.input_resolution))

        for name, module in self.backbone.named_modules():
            if name in self.target_layer_names:
                module.register_forward_hook(self.save_features_hook(name))
        
        self.face_decoder = FaceDecoder(
            transformer_dim=256,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ))    
        
        num_encoder_blocks = 4
        if self.model in ["swin_base", "swinv2_base", "convnext_base"]:
            hidden_sizes = [128, 256, 512, 1024] ### Swin Base and ConvNext Base
        if self.model in ["resnet"]:
            hidden_sizes = [256, 512, 1024, 2048] ### ResNet
        if self.model in ["swinv2_small", "swinv2_tiny", "convnext_small", "convnext_tiny"]:
            hidden_sizes = [96, 192, 384, 768] ### Swin Small/Tiny and ConvNext Small/Tiny
        if self.model in ["mobilenet"]:
            hidden_sizes = [24, 40, 112, 960] ### MobileNet
        if self.model in ["efficientnet"]:
            hidden_sizes = [48, 80, 176, 1280] ### EfficientNet
        decoder_hidden_size = 256
        
        mlps = []
        for i in range(num_encoder_blocks):
            mlp = SegfaceMLP(input_dim=hidden_sizes[i])
            mlps.append(mlp)
        self.linear_c = nn.ModuleList(mlps)

        # The following 3 layers implement the ConvModule of the original implementation
        self.linear_fuse = nn.Conv2d(
            in_channels=decoder_hidden_size * num_encoder_blocks,
            out_channels=decoder_hidden_size,
            kernel_size=1,
            bias=False,
        )

    
    def save_features_hook(self, name):
        def hook(module, input, output):
            if self.model in ["swin_base", "swinv2_base", "swinv2_small", "swinv2_tiny"]:
                self.multi_scale_features.append(output.permute(0,3,1,2).contiguous()) ### Swin, Swinv2
            if self.model in ["convnext_base", "convnext_small", "convnext_tiny", "mobilenet", "efficientnet"]:
                self.multi_scale_features.append(output) ### ConvNext, ResNet, EfficientNet, MobileNet
        return hook

    def forward(self, x, labels=None, dataset='Lapa'):
        self.multi_scale_features.clear()
        
        _,_,h,w = x.shape
        
        if labels is not None:
            mask = dataset == 1
            if mask.any(): 
                x_seg = x[mask]
                lnd = labels["lnm_seg"][mask]
                matrix = self.get_matrix_fn(lnd)
                grid = self.warp_fn(matrix=matrix, orig_shape=(h, w))
                inv_grid = self.inv_warp_fn(matrix=matrix, orig_shape=(h, w))
                w_x_seg = F.grid_sample(x_seg, grid, mode='bilinear', align_corners=False)
                x = x.clone() 
                x[mask] = w_x_seg

        features = self.backbone(x).squeeze()
        
        # batch_size = self.multi_scale_features[-1].shape[0]
        # all_hidden_states = ()
        # for encoder_hidden_state, mlp in zip(self.multi_scale_features, self.linear_c):
        #     height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
        #     encoder_hidden_state = mlp(encoder_hidden_state)
        #     encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)
        #     encoder_hidden_state = encoder_hidden_state.reshape(batch_size, -1, height, width)
        #     encoder_hidden_state = nn.functional.interpolate(
        #         encoder_hidden_state, size=self.multi_scale_features[0].size()[2:], 
        #         mode="bilinear", align_corners=False
        #     )
        #     all_hidden_states += (encoder_hidden_state,)
        
        # fused_states = self.linear_fuse(torch.cat(all_hidden_states[::-1], dim=1))
        # image_pe = self.pe_layer((fused_states.shape[2], fused_states.shape[3])).unsqueeze(0)
        # seg_output = self.face_decoder(
        #     image_embeddings=fused_states,
        #     image_pe=image_pe
        # )


        # if labels is not None and dataset is not None:
        #     segmentation_indices = dataset == 1
        #     if segmentation_indices.any():
        #         seg_lapa_in = seg_output[segmentation_indices]
        #         seg_lapa = F.grid_sample(seg_lapa_in, inv_grid, mode='bilinear', align_corners=False)
        #         seg_output = seg_output.clone()
        #         seg_output[segmentation_indices] = seg_lapa
        #         return seg_output[segmentation_indices]
        
        return self.multi_scale_features[3]

if __name__ == "__main__":
    input_resolution = 512
    model_name = "swin_base"
    model = SegFaceLapa_infer(input_resolution, model_name)
    
    batch_size = 4
    num_channels = 3
    height = input_resolution
    width = input_resolution

    x = torch.randn(batch_size, num_channels, height, width)
    
    labels = {
        "lnm_seg": torch.randn(batch_size, 5, 2)
    }

    dataset = torch.tensor([0,0,1,1])

    seg_output = model(x, labels, dataset)
    print("Segmentation Output Shape:", seg_output.shape)