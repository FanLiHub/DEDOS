import math
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from scipy.optimize import linear_sum_assignment


class SinePositionalEncoding(nn.Module):


    def __init__(
        self,
        num_feats: int = 128,
        temperature: int = 10000,
        normalize: bool = True,
        scale: Optional[float] = None,
    ):
        super().__init__()
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and not normalize:
            raise ValueError("If scale is passed, normalize should be True.")
        self.scale = scale if scale is not None else 2 * math.pi

    def forward(self, mask: Tensor) -> Tensor:

        assert mask.dim() == 3
        not_mask = ~mask

        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(
            self.num_feats, dtype=torch.float32, device=mask.device
        )
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_feats)

        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t

        pos_x = torch.stack(
            (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=4
        ).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class SimplePixelDecoder(nn.Module):

    def __init__(
        self,
        in_channels: List[int],
        feat_channels: int,
        out_channels: int,
    ):
        super().__init__()
        c_last = in_channels[-1]
        self.input_proj = nn.Conv2d(c_last, feat_channels, kernel_size=1)
        self.mask_proj = nn.Conv2d(c_last, out_channels, kernel_size=1)

    def forward(
        self, x: Tuple[Tensor]
    ) -> Tuple[Tensor, Tensor]:

        c5 = x[-1]
        memory = self.input_proj(c5)
        mask_features = self.mask_proj(c5)
        return mask_features, memory


class DetrTransformerDecoderLayer(nn.Module):

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        feedforward_dims: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=dropout, batch_first=True
        )

        self.linear1 = nn.Linear(embed_dims, feedforward_dims)
        self.linear2 = nn.Linear(feedforward_dims, embed_dims)

        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        query_pos: Optional[Tensor] = None,
        key_pos: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:

        q = query + (query_pos if query_pos is not None else 0)
        q2, _ = self.self_attn(q, q, q, need_weights=False)
        query = query + self.dropout1(q2)
        query = self.norm1(query)

        k = key + (key_pos if key_pos is not None else 0)
        q_cross = query + (query_pos if query_pos is not None else 0)
        q2, _ = self.cross_attn(
            q_cross,
            k,
            value,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        query = query + self.dropout2(q2)
        query = self.norm2(query)

        q2 = self.linear2(self.dropout_ffn(self.activation(self.linear1(query))))
        query = query + self.dropout3(q2)
        query = self.norm3(query)
        return query


class DetrTransformerDecoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 6,
        embed_dims: int = 256,
        num_heads: int = 8,
        feedforward_dims: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            DetrTransformerDecoderLayer(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_dims=feedforward_dims,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.embed_dims = embed_dims

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        query_pos: Optional[Tensor] = None,
        key_pos: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        output = query
        intermediate = []
        for layer in self.layers:
            output = layer(
                query=output,
                key=key,
                value=value,
                query_pos=query_pos,
                key_pos=key_pos,
                key_padding_mask=key_padding_mask,
            )
            intermediate.append(output)
        return torch.stack(intermediate, dim=0)



def dice_loss(
    prob: Tensor,
    target: Tensor,
    eps: float = 1e-5,
) -> Tensor:

    prob = prob.flatten(1)
    target = target.flatten(1)
    intersection = (prob * target).sum(-1)
    union = prob.sum(-1) + target.sum(-1)
    dice = (2 * intersection + eps) / (union + eps)
    return (1 - dice).mean()


def sigmoid_focal_loss(
    logits: Tensor,
    targets: Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    eps: float = 1e-6,
) -> Tensor:

    targets = targets.float()
    prob = logits.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )

    p_t = prob * targets + (1 - prob) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * (1 - p_t) ** gamma * ce_loss
    return loss.mean()


class Mask2FormerHeadDEDOS(nn.Module):

    def __init__(
        self,
        in_channels: List[int],
        feat_channels: int,
        out_channels: int,
        num_classes: int,
        num_queries: int = 100,
        num_transformer_layers: int = 6,
        ignore_index: int = 255,
        replace_query_feat: bool = False,
        use_external_query: bool = True,
        text_embed_dim: int = 512,
        loss_cls_weight: float = 1.0,
        loss_mask_weight: float = 20.0,
        loss_dice_weight: float = 1.0,
        cls_cost_weight: float = 1.0,
        mask_cost_weight: float = 1.0,
        dice_cost_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.feat_channels = feat_channels
        self.out_channels = out_channels
        self.ignore_index = ignore_index

        self.loss_cls_weight = loss_cls_weight
        self.loss_mask_weight = loss_mask_weight
        self.loss_dice_weight = loss_dice_weight

        self.cls_cost_weight = cls_cost_weight
        self.mask_cost_weight = mask_cost_weight
        self.dice_cost_weight = dice_cost_weight

        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        self.pixel_decoder = SimplePixelDecoder(
            in_channels=in_channels,
            feat_channels=feat_channels,
            out_channels=out_channels,
        )

        self.transformer_decoder = DetrTransformerDecoder(
            num_layers=num_transformer_layers,
            embed_dims=feat_channels,
            num_heads=8,
            feedforward_dims=2048,
            dropout=0.1,
        )

        self.decoder_pe = SinePositionalEncoding(
            num_feats=feat_channels // 2, normalize=True
        )

        self.query_pos_embed = nn.Embedding(num_queries, feat_channels)


        if self.replace_query_feat:
            self.querys2feat = nn.Linear(feat_channels, feat_channels)
        else:
            self.query_feat = nn.Embedding(num_queries, feat_channels)

        self.text_mlp = nn.Sequential(
            nn.Linear(text_embed_dim, feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels),
        )

        self.cls_attn1 = nn.MultiheadAttention(
            embed_dim=feat_channels, num_heads=8, batch_first=True
        )
        self.cls_attn2 = nn.MultiheadAttention(
            embed_dim=feat_channels, num_heads=8, batch_first=True
        )

        self.mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels, out_channels),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @torch.no_grad()
    def _build_padding_mask(self, x: Tensor) -> Tensor:
        B, _, H, W = x.shape
        return x.new_zeros((B, H, W), dtype=torch.bool)

    def forward(
        self,
        x: Tuple[Tuple[Tensor], Tensor, Tensor],
    ) -> Tuple[Tensor, Tensor]:

        features, query_embed, text_embed = x
        batch_size = features[0].shape[0]

        mask_features, memory_4d = self.pixel_decoder(features)

        padding_mask = self._build_padding_mask(memory_4d)
        pos_embed = self.decoder_pe(padding_mask)

        B, C, Hm, Wm = memory_4d.shape
        memory = memory_4d.flatten(2).permute(0, 2, 1)
        pos_embed_seq = pos_embed.flatten(2).permute(0, 2, 1)
        key_padding_mask = padding_mask.flatten(1)

        if query_embed is None:
            query_pos = self.query_pos_embed.weight.unsqueeze(0).expand(
                batch_size, -1, -1
            )
        else:
            if query_embed.dim() == 2:
                query_pos = query_embed.unsqueeze(0).expand(batch_size, -1, -1)
            elif query_embed.dim() == 3:
                assert query_embed.shape[0] == batch_size
                query_pos = query_embed
            else:
                raise ValueError(
                    f"query_embed must be (Q,C) or (B,Q,C), got {query_embed.shape}"
                )

        if self.replace_query_feat:
            query_feat = self.querys2feat(query_pos)
        else:
            query_feat = self.query_feat.weight.unsqueeze(0).expand(
                batch_size, -1, -1
            )

        out_dec = self.transformer_decoder(
            query=query_feat,
            key=memory,
            value=memory,
            query_pos=query_pos,
            key_pos=pos_embed_seq,
            key_padding_mask=key_padding_mask,
        )


        text_feat = self.text_mlp(text_embed)

        text_feat_batch = text_feat.unsqueeze(0).expand(batch_size, -1, -1)

        cls_scores_per_layer = []

        for l in range(out_dec.shape[0]):
            dec_l = out_dec[l]

            attn_out, _ = self.cls_attn1(
                dec_l,
                text_feat_batch,
                text_feat_batch,
                need_weights=False,
            )

            attn_out, _ = self.cls_attn2(
                attn_out,
                text_feat_batch,
                text_feat_batch,
                need_weights=False,
            )

            logits = torch.einsum("bqc,nc->bqn", attn_out, text_feat)

            cls_scores_per_layer.append(logits.unsqueeze(0))


        all_cls_scores = torch.cat(cls_scores_per_layer, dim=0)

        mask_embed = self.mask_embed(out_dec)
        all_mask_preds = torch.einsum(
            "lbqc,bchw->lbqhw", mask_embed, mask_features
        )

        return all_cls_scores, all_mask_preds




    @torch.no_grad()
    def _seg_to_instances(
        self,
        gt_sem_seg: Tensor,
    ):
        batch_gt_labels = []
        batch_gt_masks = []

        B, H, W = gt_sem_seg.shape
        for b in range(B):
            seg = gt_sem_seg[b]
            classes = torch.unique(seg)
            classes = classes[classes != self.ignore_index]

            gt_labels = []
            masks = []
            for c in classes:
                gt_labels.append(c)
                masks.append(seg == c)

            if len(masks) == 0:
                gt_labels = seg.new_zeros((0,), dtype=torch.long)
                gt_masks = seg.new_zeros((0, H, W), dtype=torch.bool)
            else:
                gt_labels = torch.stack(gt_labels).long()
                gt_masks = torch.stack(masks)

            batch_gt_labels.append(gt_labels)
            batch_gt_masks.append(gt_masks)

        return batch_gt_labels, batch_gt_masks


    @torch.no_grad()
    def _match_single(
        self,
        cls_scores: Tensor,
        mask_preds: Tensor,
        gt_labels: Tensor,
        gt_masks: Tensor,
    ):

        device = cls_scores.device
        num_queries = cls_scores.shape[0]
        num_gt = gt_labels.shape[0]

        if num_gt == 0:
            return (
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
            )

        H_p, W_p = mask_preds.shape[-2:]
        if gt_masks.shape[-2:] != (H_p, W_p):
            gt_masks_resized = F.interpolate(
                gt_masks.unsqueeze(1).float(),
                size=(H_p, W_p),
                mode="nearest",
            ).squeeze(1)
        else:
            gt_masks_resized = gt_masks.float()

        cls_prob = cls_scores.softmax(-1)
        cls_prob_fg = cls_prob[:, gt_labels]
        cost_cls = -cls_prob_fg

        mask_prob = mask_preds.sigmoid().flatten(1)
        gt_masks_flat = gt_masks_resized.flatten(1)

        eps = 1e-6
        inter = (mask_prob.unsqueeze(1) * gt_masks_flat.unsqueeze(0)).sum(-1)
        sum_p = mask_prob.sum(-1).unsqueeze(1)
        sum_g = gt_masks_flat.sum(-1).unsqueeze(0)
        dice = (2 * inter + eps) / (sum_p + sum_g + eps)
        cost_dice = 1 - dice


        mask_prob_expand = mask_prob.unsqueeze(1)
        gt_expand = gt_masks_flat.unsqueeze(0)
        bce = (
            -(gt_expand * (mask_prob_expand + eps).log())
            - ((1 - gt_expand) * (1 - mask_prob_expand + eps).log())
        ).mean(-1)
        cost_mask = bce

        cost = (
            self.cls_cost_weight * cost_cls
            + self.dice_cost_weight * cost_dice
            + self.mask_cost_weight * cost_mask
        )

        q_ind, g_ind = linear_sum_assignment(cost.detach().cpu().numpy())
        q_ind = torch.as_tensor(q_ind, dtype=torch.long, device=device)
        g_ind = torch.as_tensor(g_ind, dtype=torch.long, device=device)
        return q_ind, g_ind


    def _loss_single_layer(
        self,
        cls_scores: Tensor,
        mask_preds: Tensor,
        gt_sem_seg: Tensor,
    ):
        B, Q, _ = cls_scores.shape
        _, _, H_p, W_p = mask_preds.shape

        batch_gt_labels, batch_gt_masks = self._seg_to_instances(gt_sem_seg)

        device = cls_scores.device
        labels = torch.full(
            (B, Q),
            fill_value=self.num_classes,
            dtype=torch.long,
            device=device,
        )
        mask_weights = torch.zeros(
            (B, Q), dtype=torch.float32, device=device
        )
        all_mask_targets = []

        for b in range(B):
            gt_labels = batch_gt_labels[b]
            gt_masks = batch_gt_masks[b]

            if gt_labels.numel() == 0:
                continue

            q_ind, g_ind = self._match_single(
                cls_scores[b],
                mask_preds[b],
                gt_labels,
                gt_masks,
            )

            if q_ind.numel() == 0:
                continue

            labels[b, q_ind] = gt_labels[g_ind]
            mask_weights[b, q_ind] = 1.0

            H_gt, W_gt = gt_masks.shape[-2:]
            if (H_gt, W_gt) != (H_p, W_p):
                gt_masks_resized = F.interpolate(
                    gt_masks[g_ind].unsqueeze(1).float(),
                    size=(H_p, W_p),
                    mode="nearest",
                ).squeeze(1)
            else:
                gt_masks_resized = gt_masks[g_ind].float()

            all_mask_targets.append(gt_masks_resized)

        cls_scores_flat = cls_scores.reshape(B * Q, -1)
        labels_flat = labels.reshape(-1)

        loss_cls = F.cross_entropy(
            cls_scores_flat, labels_flat, reduction="mean"
        )

        if len(all_mask_targets) == 0:
            loss_mask = cls_scores_flat.new_tensor(0.0)
            loss_dice = cls_scores_flat.new_tensor(0.0)
        else:
            mask_targets = torch.cat(all_mask_targets, dim=0)
            mask_preds_pos = mask_preds[mask_weights > 0]

            loss_dice = dice_loss(
                mask_preds_pos.sigmoid(), mask_targets
            )
            loss_mask = sigmoid_focal_loss(
                mask_preds_pos,
                mask_targets,
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
            )

        return (
            self.loss_cls_weight * loss_cls,
            self.loss_mask_weight * loss_mask,
            self.loss_dice_weight * loss_dice,
        )

    def loss(
        self,
        x: Tuple[Tensor],
        gt_sem_seg: Tensor,
    ) -> Dict[str, Tensor]:

        all_cls_scores, all_mask_preds = self.forward(x)
        num_layers = all_cls_scores.shape[0]

        loss_cls_list = []
        loss_mask_list = []
        loss_dice_list = []

        for l in range(num_layers):
            cls_l = all_cls_scores[l]
            mask_l = all_mask_preds[l]
            lc, lm, ld = self._loss_single_layer(cls_l, mask_l, gt_sem_seg)
            loss_cls_list.append(lc)
            loss_mask_list.append(lm)
            loss_dice_list.append(ld)

        loss_dict = {
            "loss_cls": loss_cls_list[-1],
            "loss_mask": loss_mask_list[-1],
            "loss_dice": loss_dice_list[-1],
        }


        for i in range(num_layers - 1):
            loss_dict[f"d{i}.loss_cls"] = loss_cls_list[i]
            loss_dict[f"d{i}.loss_mask"] = loss_mask_list[i]
            loss_dict[f"d{i}.loss_dice"] = loss_dice_list[i]

        return loss_dict


    @torch.no_grad()
    def predict(
        self,
        x: Tuple[Tensor],
        img_shape: Tuple[int, int],
    ) -> Tensor:

        all_cls_scores, all_mask_preds = self.forward(x)
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]

        H, W = img_shape
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )

        cls_score = F.softmax(mask_cls_results, dim=-1)[..., :-1]
        mask_pred = mask_pred_results.sigmoid()

        seg_logits = torch.einsum("bqc,bqhw->bchw", cls_score, mask_pred)
        return seg_logits
