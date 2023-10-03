from typing import Dict

import torch
from torch import nn
from yacs.config import CfgNode


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    # https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer.generate_square_subsequent_mask
    mask = ~(torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    return mask


class CategoryBoxEmbeddings(nn.Module):
    def __init__(self):
        super(CategoryBoxEmbeddings, self).__init__()
        # Fixing params
        self.hidden_size = 768
        self.hidden_dropout_prob = 0.1
        self.layer_norm_eps = 0.1
        # [Hand, Object, Padding]
        self.category_embeddings = nn.Embedding(
            embedding_dim=self.hidden_size, num_embeddings=3, padding_idx=0
        )
        self.box_embedding = nn.Linear(4, self.hidden_size)
        self.score_embedding = nn.Linear(1, self.hidden_size)
        # [Pad, Left, Right]
        self.side_embedding = nn.Embedding(
            num_embeddings=3, embedding_dim=self.hidden_size, padding_idx=0
        )
        # [Pad, No contact, Self contact, Another person, Portable obj., Stationary obj.]
        self.state_embedding = nn.Embedding(
            num_embeddings=6, embedding_dim=self.hidden_size, padding_idx=0
        )
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        category_embeddings = self.category_embeddings(batch["class_labels"])
        boxes_embeddings = self.box_embedding(batch["bboxes"])
        score_embeddings = self.score_embedding(batch["scores"].unsqueeze(-1))
        sides_embedding = self.side_embedding(batch["sides"])
        states_embedding = self.state_embedding(batch["states"])
        embeddings = (
            category_embeddings
            + boxes_embeddings
            + score_embeddings
            + sides_embedding
            + states_embedding
        )
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class SpatialTransformer(nn.Module):
    def __init__(self):
        super(SpatialTransformer, self).__init__()
        # Fixing params
        self.hidden_size = 768
        self.hidden_dropout_prob = 0.1
        self.num_attention_heads = 8
        self.num_layers = 6
        # Rest
        self.category_box_embeddings = CategoryBoxEmbeddings()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=self.num_attention_heads,
                dim_feedforward=self.hidden_size * 4,
                dropout=self.hidden_dropout_prob,
                activation="gelu",
            ),
            num_layers=self.num_layers,
        )

    def forward(self, batch: Dict[str, torch.Tensor]):
        # [Batch size, Num. frames, Num. boxes, Hidden size]
        cb_embeddings = self.category_box_embeddings(batch)
        bs, nf, nb, hs = cb_embeddings.size()
        # Add CLS token
        cb_embeddings = torch.cat(
            (self.cls_token.expand(bs, nf, -1, -1), cb_embeddings), dim=2
        )
        src_key_padding_mask_boxes = torch.cat(
            (
                torch.zeros(bs, nf, 1, dtype=torch.bool, device=cb_embeddings.device),
                batch["src_key_padding_mask_boxes"],
            ),
            dim=2,
        )
        # [Batch size * Num. frames,  Num. boxes, Hidden size]
        cb_embeddings = cb_embeddings.flatten(0, 1)
        src_key_padding_mask_boxes = src_key_padding_mask_boxes.flatten(0, 1)
        # [Num. boxes, Batch size * Num. frames, Hidden size]
        cb_embeddings = cb_embeddings.transpose(0, 1)
        # [Num. boxes, Batch size * Num. frames, Hidden size]
        layout_embeddings = self.transformer(
            src=cb_embeddings,
            src_key_padding_mask=src_key_padding_mask_boxes,
        )
        # [Batch size * Num. frames, Num. boxes, Hidden size]
        layout_embeddings = layout_embeddings.transpose(0, 1)
        # [Batch size, Num. frames, Num. boxes, Hidden size]
        layout_embeddings = layout_embeddings.view(bs, nf, nb + 1, hs)
        # [Batch size, Num. frames, Hidden size]
        layout_embeddings = layout_embeddings[:, :, 0, :]

        return layout_embeddings


class TemporalTransformer(nn.Module):
    def __init__(self, cfg: CfgNode):
        super(TemporalTransformer, self).__init__()
        # Fixing params
        self.hidden_size = 768
        self.hidden_dropout_prob = 0.1
        self.layer_norm_eps = 0.1
        self.num_attention_heads = 8
        self.num_layers = 6
        self.num_frames = cfg.NUM_FRAMES
        # Rest
        self.layout_embedding = SpatialTransformer()
        self.position_embeddings = nn.Embedding(self.num_frames, self.hidden_size)
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        # Temporal Transformer
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=self.num_attention_heads,
                dim_feedforward=self.hidden_size * 4,
                dropout=self.hidden_dropout_prob,
                activation="gelu",
            ),
            num_layers=self.num_layers,
        )

    def forward(self, batch: Dict[str, torch.Tensor]):
        # [Batch size, Num. frames, Hidden size]
        layout_embeddings = self.layout_embedding(batch)
        bs, nf, _ = layout_embeddings.size()
        position_embeddings = self.position_embeddings(
            torch.arange(nf, device=layout_embeddings.device).expand(1, -1)
        )
        # Preparing everything together
        embeddings = layout_embeddings + position_embeddings
        embeddings = self.dropout(self.layer_norm(embeddings))
        # Concatenate with CLS token
        embeddings = torch.cat((embeddings, self.cls_token.expand(bs, -1, -1)), dim=1)
        # [Num. frames, Batch size, Hidden size]
        embeddings = embeddings.transpose(0, 1)
        # [Num. frames, Batch size, Hidden size]
        causal_mask = generate_square_subsequent_mask(embeddings.size(0)).to(
            embeddings.device
        )
        layout_embeddings = self.transformer(src=embeddings, mask=causal_mask)
        # [Batch size, Hidden size]
        layout_embeddings = layout_embeddings[-1, :, :]
        # Make contiguous
        layout_embeddings = layout_embeddings.contiguous()

        return layout_embeddings


class Stlt(nn.Module):
    def __init__(self, cfg: CfgNode):
        super(Stlt, self).__init__()
        # Fixing params
        self.cfg = cfg
        self.hidden_size = 768
        # Rest
        self.temporal_transformer = TemporalTransformer(self.cfg)
        # Build classifier
        self.classifiers = nn.ModuleDict(
            {
                actions_name: nn.Linear(self.hidden_size, actions_num)
                for actions_name, actions_num in self.cfg.TOTAL_ACTIONS.items()
                if actions_num is not None
            }
        )

    def forward(self, batch: Dict[str, torch.Tensor]):
        output = {}
        # Get features
        features = self.temporal_transformer(batch)
        for actions_name in self.classifiers.keys():
            output[actions_name] = self.classifiers[actions_name](features)

        return output
