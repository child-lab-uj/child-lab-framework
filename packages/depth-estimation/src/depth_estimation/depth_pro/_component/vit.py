import types
from dataclasses import dataclass, field
from typing import Literal, cast

import timm
import torch
import torch.nn as nn
from timm.layers.pos_embed import resample_abs_pos_embed
from torch.utils.checkpoint import checkpoint

# +---------------------------------------------------------------------+
# | Code adopted from:                                                  |
# | Repository: https://github.com/apple/ml-depth-pro                   |
# | Commit: b2cd0d51daa95e49277a9f642f7fd736b7f9e91d                    |
# | Files:                                                              |
# |   - `src/depth_pro/network/vit.py`                                  |
# |   - `src/depth_pro/network/vit_factory.py`                          |
# | Acknowledgement: Copyright (C) 2024 Apple Inc. All Rights Reserved. |
# +---------------------------------------------------------------------+


type Preset = Literal['dinov2l16_384',]


@dataclass
class Configuration:
    """Configuration for ViT."""

    input_channels: int
    embedding_size: int

    image_size: int = 384
    patch_size: int = 16

    # In case we need to rescale the backbone when loading from timm.
    timm_preset: str = 'vit_large_patch14_dinov2'
    timm_image_size: int = 384
    timm_patch_size: int = 16

    # The following 2 parameters are only used by DPT.  See dpt_factory.py.
    encoder_feature_layer_ids: list[int] = field(default_factory=list)
    """The layers in the Beit/ViT used to constructs encoder features for DPT."""
    encoder_feature_dims: list[int] = field(default_factory=list)
    """The dimension of features of encoder layers from Beit/ViT features for DPT."""


PRESETS: dict[Preset, Configuration] = {
    'dinov2l16_384': Configuration(
        input_channels=3,
        embedding_size=1024,
        encoder_feature_layer_ids=[5, 11, 17, 23],
        encoder_feature_dims=[256, 512, 1024, 1024],
        image_size=384,
        patch_size=16,
        timm_preset='vit_large_patch14_dinov2',
        timm_image_size=518,
        timm_patch_size=14,
    ),
}


def create(
    preset: Preset,
    use_pretrained: bool = False,
    checkpoint_uri: str | None = None,
    use_grad_checkpointing: bool = False,
) -> nn.Module:
    """
    Create and load a VIT backbone module.

    Parameters
    ---
    preset: Preset
        The VIT preset to load the pre-defined config.

    use_pretrained: bool
        Load pretrained weights if True, default is False.

    checkpoint_uri: str | None
        Checkpoint to load the wights from.

    use_grad_checkpointing: bool
        Use gradient checkpointing.

    Returns
    ---
    result: torch.nn.Module
        A Torch ViT backbone module.
    """

    configuration = PRESETS[preset]

    img_size = (configuration.image_size, configuration.image_size)
    patch_size = (configuration.patch_size, configuration.patch_size)

    if 'eva02' in preset:
        model = timm.create_model(configuration.timm_preset, pretrained=use_pretrained)
        model.forward_features = types.MethodType(forward_features_eva_fixed, model)
    else:
        model = timm.create_model(
            configuration.timm_preset,
            pretrained=use_pretrained,
            dynamic_img_size=True,
        )

    model = vit_b16_backbone(
        model,
        encoder_feature_dims=configuration.encoder_feature_dims,
        encoder_feature_layer_ids=configuration.encoder_feature_layer_ids,
        vit_features=configuration.embedding_size,
        use_grad_checkpointing=use_grad_checkpointing,
    )

    if configuration.patch_size != configuration.timm_patch_size:
        model.model = resize_patch_embed(
            cast(nn.Module, model.model),
            new_patch_size=patch_size,
        )

    if configuration.image_size != configuration.timm_image_size:
        model.model = resize_vit(cast(nn.Module, model.model), img_size=img_size)

    if checkpoint_uri is not None:
        state_dict = torch.load(checkpoint_uri, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(
            state_dict=state_dict, strict=False
        )

        if len(unexpected_keys) != 0:
            raise KeyError(f'Found unexpected keys when loading vit: {unexpected_keys}')
        if len(missing_keys) != 0:
            raise KeyError(f'Keys are missing when loading vit: {missing_keys}')

    return cast(nn.Module, model.model)


# TODO: Write a custom class with typed fields and remove this awful prototyping.
def vit_b16_backbone(
    model: timm.models.vision_transformer.VisionTransformer,
    encoder_feature_dims: list[int],
    encoder_feature_layer_ids: list[int],
    vit_features: int,
    start_index: int = 1,
    use_grad_checkpointing: bool = False,
) -> nn.Module:
    """Make a ViTb16 backbone for the DPT model."""

    if use_grad_checkpointing:
        model.set_grad_checkpointing()

    # Ignores make prototyping possible.
    vit_model = nn.Module()
    vit_model.hooks = encoder_feature_layer_ids  # type: ignore
    vit_model.model = model
    vit_model.features = encoder_feature_dims  # type: ignore
    vit_model.vit_features = vit_features  # type: ignore
    vit_model.model.start_index = start_index  # type: ignore
    vit_model.model.patch_size = vit_model.model.patch_embed.patch_size
    vit_model.model.is_vit = True  # type: ignore
    vit_model.model.forward = vit_model.model.forward_features  # type: ignore[method-assign]

    return vit_model


def forward_features_eva_fixed(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Encode features."""

    x = model.patch_embed(x)  # type: ignore
    x, rot_pos_embed = model._pos_embed(x)  # type: ignore

    blocks: list[nn.Module] = model.blocks  # type: ignore
    for block in blocks:
        if model.grad_checkpointing:
            x = cast(torch.Tensor, checkpoint(block, x, rot_pos_embed))
        else:
            x = block(x, rot_pos_embed)

    x = model.norm(x)  # type: ignore

    return x


def resize_vit(model: nn.Module, img_size: tuple[int, int]) -> nn.Module:
    """Resample the ViT module to the given size."""

    patch_size: tuple[int, int] = model.patch_embed.patch_size  # type: ignore
    model.patch_embed.img_size = img_size  # type: ignore

    grid_size = [
        img_size[0] // patch_size[0],
        img_size[1] // patch_size[1],
    ]
    model.patch_embed.grid_size = grid_size  # type: ignore

    num_prefix_tokens: int = cast(
        int, 0 if getattr(model, 'no_embed_class', False) else model.num_prefix_tokens
    )
    pos_embed = resample_abs_pos_embed(
        cast(torch.Tensor, model.pos_embed),
        new_size=grid_size,  # img_size (idk what does this comment mean, I left it here long ago ;v).
        num_prefix_tokens=num_prefix_tokens,
    )
    model.pos_embed = torch.nn.Parameter(pos_embed)

    return model


def resize_patch_embed(
    model: nn.Module,
    new_patch_size: tuple[int, int] = (16, 16),
) -> nn.Module:
    """Resample the ViT patch size to the given one."""

    # interpolate patch embedding

    if not hasattr(model, 'patch_embed'):
        return model

    old_patch_size: tuple[int, int] = model.patch_embed.patch_size  # type: ignore

    if new_patch_size[0] == old_patch_size[0] and new_patch_size[1] == old_patch_size[1]:
        return model

    patch_embed_proj: nn.Parameter = model.patch_embed.proj.weight  # type: ignore
    patch_embed_proj_bias: nn.Parameter = model.patch_embed.proj.bias  # type: ignore

    use_bias = True if patch_embed_proj_bias is not None else False

    h: int
    w: int
    _, _, h, w = patch_embed_proj.shape

    new_patch_embed_proj = torch.nn.functional.interpolate(
        patch_embed_proj,
        size=[new_patch_size[0], new_patch_size[1]],
        mode='bicubic',
        align_corners=False,
    )
    new_patch_embed_proj = (
        new_patch_embed_proj * (h / new_patch_size[0]) * (w / new_patch_size[1])
    )

    in_channels: int = model.patch_embed.proj.in_channels  # type: ignore
    out_channels: int = model.patch_embed.proj.out_channels  # type: ignore

    model.patch_embed.proj = nn.Conv2d(  # type: ignore
        in_channels,
        out_channels,
        kernel_size=new_patch_size,
        stride=new_patch_size,
        bias=use_bias,
    )

    if use_bias:
        model.patch_embed.proj.bias = patch_embed_proj_bias  # type: ignore

    model.patch_embed.proj.weight = (  # type: ignore
        torch.nn.Parameter(new_patch_embed_proj)
    )

    img_height: int = model.patch_embed.img_size[0]  # type: ignore
    img_width: int = model.patch_embed.img_size[1]  # type: ignore

    model.patch_size = new_patch_size  # type: ignore
    model.patch_embed.patch_size = new_patch_size  # type: ignore
    model.patch_embed.img_size = (  # type: ignore
        int(img_height * new_patch_size[0] / old_patch_size[0]),
        int(img_width * new_patch_size[1] / old_patch_size[1]),
    )

    return model
