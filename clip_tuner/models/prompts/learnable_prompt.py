from dataclasses import dataclass
from typing import Any, Optional
from collections.abc import Mapping

import torch
from torch import nn
from transformers.models.clip.modeling_clip import (
    CLIPModel,
    CLIPTextConfig,
    CLIPTextEmbeddings,
)


class CLIPTextLearnableEmbeddings(CLIPTextEmbeddings):
    def __init__(
        self,
        config: CLIPTextConfig,
        n_ctx: Optional[int] = 2,
        start_idx: Optional[
            int
        ] = 1,  # 默认"[SOS] a photo of a [CLS].[EOS]"，0 是 [SOS]
    ):
        super().__init__(config=config)
        embed_dim = config.hidden_size
        self.learnable_embeddings = nn.Parameter(
            nn.init.normal_(torch.empty([n_ctx, embed_dim]), std=0.02)
        )

        self.n_ctx = n_ctx
        self.start_idx = start_idx
        self.end_idx = start_idx + n_ctx

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        seq_length = (
            input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]
        )
        max_position_embedding = self.position_embedding.weight.shape[0]

        if seq_length > max_position_embedding:
            raise ValueError(
                f"Sequence length must be less than max_position_embeddings (got `sequence length`: "
                f"{seq_length} and max_position_embeddings: {max_position_embedding}"
            )

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        batch_learnable_embeddings = self.learnable_embeddings.unsqueeze(0).expand(
            input_ids.shape[0], -1, -1
        )

        inputs_embeds = torch.cat(
            [
                inputs_embeds[:, : self.start_idx],  # prefix: [SOS]
                batch_learnable_embeddings,  # add learnable_prompts
                inputs_embeds[:, self.end_idx :],  # suffix
            ],
            dim=1,
        )

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings


def add_learnable_prompts_to_clip_text_model(
    clip_model: CLIPModel,
    n_ctx: Optional[int] = 2,
    start_idx: Optional[int] = 1,
    learnable_embeddings_state_dict: Optional[Mapping[str, Any]] = None,
) -> CLIPModel:
    # 获取原始文本嵌入配置
    text_config = clip_model.text_model.config
    learnable_embeddings = CLIPTextLearnableEmbeddings(
        text_config, n_ctx=n_ctx, start_idx=start_idx
    )
    if learnable_embeddings_state_dict is not None:
        learnable_embeddings.load_state_dict(learnable_embeddings_state_dict)
    else:
        learnable_embeddings.token_embedding = clip_model.text_model.embeddings
        learnable_embeddings.position_embedding = (
            clip_model.text_model.embeddings.position_embedding
        )
    clip_model.text_model.embeddings = learnable_embeddings
    return clip_model
