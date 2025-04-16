from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision.transforms import (
    Compose,
    ConvertImageDtype,
    Normalize,
)

from ._component import vit
from ._component.decoder import Decoder
from ._component.encoder import Encoder
from ._component.fov import FovNetwork

# +---------------------------------------------------------------------+
# | Code adopted from:                                                  |
# | Repository: https://github.com/apple/ml-depth-pro                   |
# | Commit: b2cd0d51daa95e49277a9f642f7fd736b7f9e91d                    |
# | File: `src/depth_pro/depth_pro.py`                                  |
# | Acknowledgement: Copyright (C) 2024 Apple Inc. All Rights Reserved. |
# +---------------------------------------------------------------------+


class Configuration:
    checkpoint: Path

    decoder_features: int
    output_shape: tuple[int, int]

    patch_encoder_preset: vit.Preset
    patch_encoder_config: vit.Configuration

    image_encoder_preset: vit.Preset
    image_encoder_config: vit.Configuration

    fov_encoder_preset: vit.Preset | None
    fov_encoder_config: vit.Configuration | None

    def __init__(
        self,
        checkpoint: Path,
        decoder_features: int = 256,
        output_shape: tuple[int, int] = (32, 1),
        patch_encoder_preset: vit.Preset = 'dinov2l16_384',
        image_encoder_preset: vit.Preset = 'dinov2l16_384',
        fov_encoder_preset: vit.Preset | None = 'dinov2l16_384',
    ) -> None:
        self.checkpoint = checkpoint
        self.decoder_features = decoder_features
        self.output_shape = output_shape

        self.patch_encoder_preset = patch_encoder_preset
        self.path_encoder_config = vit.PRESETS[patch_encoder_preset]

        self.image_encoder_preset = image_encoder_preset
        self.image_encoder_config = vit.PRESETS[image_encoder_preset]

        self.fov_encoder_preset = fov_encoder_preset
        self.fov_encoder_config = (
            vit.PRESETS[fov_encoder_preset] if fov_encoder_preset is not None else None
        )


@dataclass
class Result:
    focal_length_px: float
    depth: torch.Tensor


class DepthPro:
    """DepthPro model."""

    config: Configuration
    device: torch.device
    dtype: torch.dtype

    input_transformation: Compose

    model: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor | None]]
    input_image_size: int
    has_fov: bool

    def __init__(
        self,
        configuration: Configuration,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """
        Initialize DepthPro.

        Parameters
        ---
        configuration: Configuration
            Configuration of the model.

        device: torch.device
            Torch device to use for tensor computations.

        dtype: torch.dtype
            Data type to use in tensor computation.
        """

        super().__init__()

        self.configuration = configuration
        self.device = device
        self.dtype = dtype

        self.input_transformation = Compose(  # type: ignore[no-untyped-call]
            [
                ConvertImageDtype(dtype),
                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # type: ignore[no-untyped-call]
            ]
        )

        model = DepthProModule(configuration).to(device)

        if dtype == torch.half:
            model.half()

        self.input_image_size = model.encoder.image_size
        self.has_fov = model.fov is not None
        self.model = model  # TODO: torch.compile

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        focal_length_px: float | None = None,
        interpolation_mode: str = 'bilinear',
    ) -> Result:
        """
        Estimate depth and field of view for a given image.

        If the image is not at network resolution, it is resized to 1536x1536 and
        the estimated depth is resized to the original image resolution.
        Note: if the focal length is given, the estimated value is ignored and the provided
        focal length is use to generate the metric depth values.

        Parameters
        ---
        x: torch.Tensor
            Input image

        focal_length_px: float | None
            Optional focal length in the x dimension, in pixels.

        interpolation_mode: str
            Interpolation function for downsampling/upsampling.

        Returns
        ---
        result: Result
        """

        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        if focal_length_px is None and self.has_fov is None:
            raise ValueError(
                'Focal length was not specified. It cannot be inferred because FOV head is off.'
            )

        input_size = self.input_image_size

        _, _, height, width = x.shape
        resize = height != input_size or width != input_size

        x = self.input_transformation(x)

        if resize:
            x = nn.functional.interpolate(
                x,
                size=(input_size, input_size),
                mode=interpolation_mode,
                align_corners=False,
            )

        canonical_inverse_depth: torch.Tensor
        fov_angle_degrees: torch.Tensor | None
        canonical_inverse_depth, fov_angle_degrees = self.model(x)

        match focal_length_px, fov_angle_degrees:
            case None, None:
                assert False, 'unreachable'

            case float(value), _:
                f_px = torch.tensor(value)

            case None, angle if angle is not None:
                fov_angle_degrees = angle.to(torch.float)
                f_px = torch.squeeze(0.5 * width / torch.tan(0.5 * torch.deg2rad(angle)))

            case _, _:
                assert False, 'unreachable'

        inverse_depth = canonical_inverse_depth * (width / f_px)

        if resize:
            inverse_depth = nn.functional.interpolate(
                inverse_depth,
                size=(height, width),
                mode=interpolation_mode,
                align_corners=False,
            )

        depth = 1.0 / torch.clamp(inverse_depth, min=1e-4, max=1e4)

        return Result(f_px.item(), depth.squeeze())


class DepthProModule(nn.Module):
    """DepthPro network."""

    encoder: Encoder
    decoder: Decoder
    head: nn.Sequential
    fov: 'FovNetwork | LooseIdentity'
    has_fov: bool

    def __init__(self, configuration: Configuration) -> None:
        """
        Initialize internal DepthPro model.

        Parameters
        ---
        configuration: Configuration
            Configuration of the model.
        """

        super().__init__()

        output_shape = configuration.output_shape

        patch_encoder_config = configuration.path_encoder_config
        fov_encoder_config = configuration.fov_encoder_config

        patch_encoder = vit.create(
            preset=configuration.patch_encoder_preset,
            use_pretrained=False,
        )

        image_encoder = vit.create(
            preset=configuration.image_encoder_preset,
            use_pretrained=False,
        )

        fov_encoder = (
            vit.create(
                preset=configuration.fov_encoder_preset,
                use_pretrained=False,
            )
            if configuration.fov_encoder_preset is not None
            else nn.Identity()
        )

        dims_encoder = patch_encoder_config.encoder_feature_dims
        hook_block_ids = patch_encoder_config.encoder_feature_layer_ids

        assert dims_encoder is not None
        assert hook_block_ids is not None

        encoder = Encoder(
            decoder_features=configuration.decoder_features,
            stage_dimensions=dims_encoder,
            hook_block_ids=hook_block_ids,
            patch_encoder=patch_encoder,
            image_encoder=image_encoder,
        )

        decoder = Decoder(
            dims_encoder=[configuration.decoder_features] + encoder.stage_dimensions,
            dim_decoder=configuration.decoder_features,
        )

        dim_decoder = decoder.dim_decoder

        head = nn.Sequential(
            nn.Conv2d(
                dim_decoder,
                dim_decoder // 2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ConvTranspose2d(
                in_channels=dim_decoder // 2,
                out_channels=dim_decoder // 2,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=True,
            ),
            nn.Conv2d(
                dim_decoder // 2,
                output_shape[0],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                output_shape[0],
                output_shape[1],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
        )
        # head[4].bias.data.fill_(0)  # type: ignore  # Is this even needed?

        fov = (
            FovNetwork(
                features=dim_decoder,
                fov_encoder=fov_encoder,
            )
            if fov_encoder_config is not None
            else LooseIdentity()
        )

        self.encoder = encoder
        self.decoder = decoder
        self.head = head
        self.fov = fov
        self.has_fov = fov_encoder_config is not None

        if configuration.checkpoint is not None:
            state_dict = torch.load(configuration.checkpoint, map_location='cpu')
            missing_keys, unexpected_keys = self.load_state_dict(
                state_dict=state_dict, strict=True
            )

            if len(unexpected_keys) != 0:
                raise KeyError(
                    f'Depth Pro checkpoint contains unexpected keys: {unexpected_keys}'
                )

            # fc_norm is only for the classification head,
            # which we would not use. We only use the encoding.
            missing_keys = [key for key in missing_keys if 'fc_norm' not in key]
            if len(missing_keys) != 0:
                raise KeyError(f'Depth Pro checkpoint is missing keys: {missing_keys}')

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Decode by projection and fusion of multi-resolution encodings.

        Parameters
        ---
        x: torch.Tensor
            Input image.

        Returns
        ---
        result: tuple[torch.Tensor, torch.Tensor | None]
            The canonical inverse depth map (in meters) and the optional estimated field of view (in degrees).
        """

        _, _, height, width = x.shape
        assert height == width == self.encoder.image_size

        encodings = self.encoder.forward(x)
        features, features_0 = self.decoder.forward(encodings)
        canonical_inverse_depth: torch.Tensor = self.head.forward(features)  # type: ignore[no-untyped-call]
        fov_deg = self.fov.forward(x, features_0.detach()) if self.has_fov else None

        return canonical_inverse_depth, fov_deg


class LooseIdentity(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Any, *args: Any) -> Any:
        return x
