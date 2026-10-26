import torch
import torch.nn.functional as F
from tensordict.tensordict import TensorDict
from torch import nn

from ArchesClimate.backbones.archesweather_layers import ICNR_init


class DCPPEncodeDecodeLayer(nn.Module):
    """
    gathers layers for the encoder and decoder
    """

    def __init__(
        self,
        img_size,
        emb_dim,
        out_emb_dim,
        patch_size,
        surface_ch,
        n_concatenated_states,
        boundary_padding=False,
        double_prev_state=False,
        total_boundary_calc=False,
        level_ch=7,
        include_orography=False,
        flatten_levels=True,
    ) -> None:
        super().__init__()
        self.__dict__.update(locals())
        self.surface_proj = nn.Conv2d(
            surface_ch, emb_dim, kernel_size=patch_size[1:], stride=patch_size[1:]
        )
        self.boundary_padding = boundary_padding
        self.pixelshuffle = nn.PixelShuffle(patch_size[-1])

        if flatten_levels:
            self.surface_deconv = nn.Conv2d(
                out_emb_dim,
                surface_ch * patch_size[-1] ** 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=0,
            )
        else:
            # separate 3D level projector; surface_ch covers surface-only channels
            level_ch_in = level_ch * n_concatenated_states
            self.level_proj = nn.Conv3d(
                level_ch_in,
                emb_dim,
                kernel_size=(1, patch_size[1], patch_size[2]),
                stride=(1, patch_size[1], patch_size[2]),
            )
            n_surface_vars = surface_ch // n_concatenated_states
            self.surface_deconv = nn.Conv2d(
                out_emb_dim,
                n_surface_vars * patch_size[-1] ** 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=0,
            )
            self.level_deconv = nn.Conv2d(
                out_emb_dim,
                level_ch * patch_size[-1] ** 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=0,
            )
            ICNR_init(
                self.level_deconv.weight,
                initializer=nn.init.kaiming_normal_,
                upscale_factor=patch_size[-1],
            )

        ICNR_init(
            self.surface_deconv.weight,
            initializer=nn.init.kaiming_normal_,
            upscale_factor=patch_size[-1],
        )

    def encode(self, state: TensorDict, cond_state: TensorDict = None):
        """
        cond state is a condition state that is concatenated to the main state
        """
        bs = state.shape[0]

        surface = state["surface"].squeeze(-3)

        if self.flatten_levels:
            level = state["level"].reshape(
                -1, state["level"].shape[-3] * state["level"].shape[-4], 143, 144
            )
            if cond_state is not None:
                cond_surface = cond_state["surface"].squeeze(-3)
                cond_level = cond_state["level"].reshape(bs, -1, 143, 144)
                surface = torch.cat([surface, cond_surface], dim=1)
                level = torch.cat([level, cond_level], dim=1)
            surface = torch.cat([surface, level], dim=1)
            if self.boundary_padding:
                surface = torch.cat(
                    [surface[:, :, :, -24:], surface, surface[:, :, :, :24]], dim=-1
                )
                surface = torch.cat(
                    [
                        torch.flip(surface[:, :, :24, :], dims=(-2, -1)),
                        surface,
                        torch.flip(surface[:, :, -24:, :], dims=(-2, -1)),
                    ],
                    dim=-2,
                )
            surface = F.pad(surface, (0, 0, 0, 1))
            surface = self.surface_proj(surface)
            return surface.unsqueeze(-3)
        else:
            # 3D path: project surface and level separately, stack along P dim
            level = state["level"]  # (B, level_ch, n_plev, H, W)
            if cond_state is not None:
                cond_surface = cond_state["surface"].squeeze(-3)
                cond_level = cond_state["level"]
                surface = torch.cat([surface, cond_surface], dim=1)
                level = torch.cat(
                    [level, cond_level], dim=1
                )  # (B, level_ch_in, n_plev, H, W)
            surface = F.pad(surface, (0, 0, 0, 1))
            surface = self.surface_proj(surface)  # (B, emb_dim, H', W')
            level = F.pad(
                level, (0, 0, 0, 1, 0, 0)
            )  # pad H: (B, level_ch_in, n_plev, H+1, W)
            level = self.level_proj(level)  # (B, emb_dim, n_plev, H', W')
            return torch.cat(
                [surface.unsqueeze(2), level], dim=2
            )  # (B, emb_dim, 1+n_plev, H', W')

    def decode(self, x):
        if not self.flatten_levels:
            # 3D path: x is (B, emb_dim, 1+n_plev, H', W')
            surface_x = x[:, :, 0, :, :]  # (B, emb_dim, H', W')
            level_x = x[:, :, 1:, :, :]  # (B, emb_dim, n_plev, H', W')

            output_surface = self.surface_deconv(surface_x)
            output_surface = self.pixelshuffle(
                output_surface
            )  # (B, n_surface_vars, H, W)
            output_surface = output_surface[:, :, :143].unsqueeze(
                -3
            )  # (B, n_surface_vars, 1, 143, W)

            B, _, n_plev, H_, W_ = level_x.shape
            level_flat = level_x.permute(0, 2, 1, 3, 4).reshape(B * n_plev, -1, H_, W_)
            output_level = self.level_deconv(level_flat)
            output_level = self.pixelshuffle(output_level)  # (B*n_plev, level_ch, H, W)
            output_level = output_level.reshape(
                B, n_plev, self.level_ch, *output_level.shape[2:]
            )
            output_level = output_level.permute(
                0, 2, 1, 3, 4
            )  # (B, level_ch, n_plev, H, W)
            output_level = output_level[:, :, :, :143, :]

            return TensorDict(
                surface=output_surface,
                level=output_level,
                batch_size=output_surface.shape[0],
            ).to(x.device)

        # 2D (flatten_levels=True) path
        plev_indices = self.level_ch * 4
        if (
            self.double_prev_state
            or self.surface_ch == 181
            or self.surface_ch == 167
            or self.surface_ch == 180
        ):
            output_dim = (self.surface_ch // 4) - plev_indices
        elif self.surface_ch == 91 or self.surface_ch == 90 or self.surface_ch == 66:
            output_dim = (self.surface_ch // 2) - plev_indices
        else:
            output_dim = (self.surface_ch // 3) - plev_indices
        surface = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4])

        output_surface = self.surface_deconv(surface)
        output_surface = self.pixelshuffle(output_surface)
        output_surface = output_surface.unsqueeze(-3)

        if self.total_boundary_calc:  # need this to be first so boundary_padding and total_boundary_calc can both be true without breaking
            output_surface_final = output_surface[:, :output_dim][
                :, :, :, : 143 + 24 + 24
            ]
            output_level = output_surface[:, output_dim : output_dim + 28].reshape(
                -1, self.level_ch, 4, 24 + 144 + 24, 24 + 144 + 24
            )[:, :, :, : 143 + 24 + 24]
        elif self.boundary_padding:
            output_surface_final = output_surface[
                :, :output_dim, :, 24 : 143 + 24, 24 : 144 + 24
            ]
            output_level = output_surface[
                :, output_dim : output_dim + 28, :, 24 : 143 + 24, 24 : 144 + 24
            ].reshape(-1, self.level_ch, 4, 143, 144)
        else:
            output_level = output_surface[
                :, output_dim : output_dim + plev_indices
            ].reshape(-1, self.level_ch, 4, 144, 144)[:, :, :, :143]
            output_surface_final = output_surface[:, :output_dim, :, :143]

        return TensorDict(
            surface=output_surface_final,
            level=output_level,
            batch_size=output_surface_final.shape[0],
        ).to(x.device)
