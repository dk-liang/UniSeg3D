from typing import Optional, List

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS

from mmengine.model import BaseModule
from mmengine.dist import get_dist_info
from mmengine.logging import MMLogger
from timm.layers import resample_abs_pos_embed

import open_clip as open_clip


@MODELS.register_module()
class OpenCLIPBackboneText(BaseModule):
    def __init__(
            self,
            model_name: str = '',
            init_cfg=None,
    ):
        assert init_cfg is not None and init_cfg['type'] == 'clip_pretrain', f"{init_cfg['type']} is not supported."
        pretrained = init_cfg['checkpoint']
        super().__init__(init_cfg=None)
        self.init_cfg = init_cfg
        self.logger = MMLogger.get_current_instance()
        rank, world_size = get_dist_info()

        if world_size > 1:
            if rank == 0:
                _ = open_clip.create_model_from_pretrained(model_name, pretrained=pretrained, return_transform=False,
                                                           logger=self.logger)
            else:
                pass
            dist.barrier()

        # Get the clip model
        clip_model = open_clip.create_model_from_pretrained(model_name, pretrained=pretrained, return_transform=False,
                                                            logger=self.logger)

        # Get the textual model
        self.text_tokenizer = open_clip.get_tokenizer(model_name)
        self.text_transformer = clip_model.transformer
        self.text_token_embedding = clip_model.token_embedding
        self.text_pe = clip_model.positional_embedding
        self.text_ln_final = clip_model.ln_final
        self.text_proj = clip_model.text_projection

        self.register_buffer('text_attn_mask', clip_model.attn_mask)

        self.param_dtype = torch.float32
        self.model_name = model_name

    def init_weights(self):
        self.logger.info(f"Init Config for {self.model_name}")
        self.logger.info(self.init_cfg)

    # adapted from
    # https://github.com/openai/CLIP/blob/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1/clip/model.py#L343
    @torch.no_grad()
    def forward(self, text_tokens):
        x = self.text_token_embedding(text_tokens).to(self.param_dtype)
        x = x + self.text_pe.to(self.param_dtype)
        x = x.permute(1, 0, 2)
        x = self.text_transformer(x, attn_mask=self.text_attn_mask)
        x = x.permute(1, 0, 2)
        x = self.text_ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)] @ self.text_proj
        return x