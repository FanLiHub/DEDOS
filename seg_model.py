import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.functional as F
import math
from functools import reduce
from operator import mul
from torch import Tensor
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils.memory import _ignore_torch_cuda_oom
import json
from einops import rearrange
from DEDOS.modeling.transformer.model import Aggregator
from DEDOS.third_party import imagenet_templates
from DenoiseDiff.diff_process import DiffusionTrainer_

from typing import Dict, List, Optional, Sequence, Tuple, Union
from .modeling.transformer.cat_seg_predictor import CATSegPredictor
from torchvision import transforms

from mask2former_head import Mask2FormerHeadDEDOS
from module import Attention
from proxy_query import object_query


@META_ARCH_REGISTRY.register()
class DEDOSModel(nn.Module):
    @configurable
    def __init__(
            self,
            *,
            size_divisibility: int,
            pixel_mean: Tuple[float],
            pixel_std: Tuple[float],
            clip_pixel_mean: Tuple[float],
            clip_pixel_std: Tuple[float],
            train_class_json: str,
            test_class_json: str,
            sliding_window: bool,
            clip_pretrained: str,
            DiffusionTrainer,
            transformer_predictor,
            query_num_layers,
            query_embed_dims,
            query_patch_sizes,
    ):

        super().__init__()
        self.size_divisibility = size_divisibility

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_mean", torch.Tensor(clip_pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_std", torch.Tensor(clip_pixel_std).view(-1, 1, 1), False)

        self.predictor = transformer_predictor
        self.sliding_window = sliding_window
        self.clip_resolution = (384, 384) if clip_pretrained == "ViT-B/16" else (336, 336)

        self.proj_dim = 768 if clip_pretrained == "ViT-B/16" else 1024
        self.fpn1 = nn.Sequential(
            nn.GroupNorm(1, self.proj_dim),
            nn.ConvTranspose2d(self.proj_dim, self.proj_dim, kernel_size=2, stride=2),
            nn.SyncBatchNorm(self.proj_dim),
            nn.GELU(),
            nn.ConvTranspose2d(self.proj_dim, self.proj_dim, kernel_size=2, stride=2),
        )

        self.fpn2 = nn.Sequential(
            nn.GroupNorm(1, self.proj_dim),
            nn.ConvTranspose2d(self.proj_dim, self.proj_dim, kernel_size=2, stride=2),
        )

        self.fpn3 = nn.GroupNorm(1, self.proj_dim)

        self.fpn4 = nn.Sequential(
            nn.GroupNorm(1, self.proj_dim), nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer_indexes = [7, 11, 15]
        self.layers = []
        for l in self.layer_indexes:
            self.predictor.clip_model.visual.transformer.resblocks[l].register_forward_hook(
                lambda m, _, o: self.layers.append(o))

        self.train_class_json = train_class_json
        self.test_class_json = test_class_json

        prompt_templates = ['A photo of a {} in the scene', ]

        self.prompt_templates = prompt_templates
        self.cache = None

        self.diffopenseg = DiffusionTrainer

        self.seg_head = Mask2FormerHeadDEDOS(len(self.test_class_texts), [1024, 1024, 1024, 1024], 1024, text_fuse=True)

        self.loss_smooth = nn.SmoothL1Loss()

        self.text_query_attn = Attention(768, 1024)
        self.query_detail = nn.Linear(256, 1024)
        self.clip_detail = nn.Linear(1024, 1024)

        self.object_query = object_query(query_num_layers, query_embed_dims, query_patch_sizes)

        for name, params in self.predictor.clip_model.named_parameters():
            params.requires_grad = False

    @classmethod
    def from_config(cls, cfg):

        return {
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "clip_pixel_mean": cfg.MODEL.CLIP_PIXEL_MEAN,
            "clip_pixel_std": cfg.MODEL.CLIP_PIXEL_STD,
            "train_class_json": cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON,
            "test_class_json": cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON,
            "sliding_window": cfg.TEST.SLIDING_WINDOW,
            "backbone_multiplier": cfg.SOLVER.BACKBONE_MULTIPLIER,
            "clip_pretrained": cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED,

            "prompt_ensemble_type": cfg.MODEL.PROMPT_ENSEMBLE_TYPE,

            "text_guidance_dim": cfg.MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_DIM,
            "text_guidance_proj_dim": cfg.MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_PROJ_DIM,
            "appearance_guidance_dim": cfg.MODEL.SEM_SEG_HEAD.APPEARANCE_GUIDANCE_DIM,
            "appearance_guidance_proj_dim": cfg.MODEL.SEM_SEG_HEAD.APPEARANCE_GUIDANCE_PROJ_DIM,

            "decoder_dims": cfg.MODEL.SEM_SEG_HEAD.DECODER_DIMS,
            "decoder_guidance_dims": cfg.MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_DIMS,
            "decoder_guidance_proj_dims": cfg.MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_PROJ_DIMS,

            "num_layers": cfg.MODEL.SEM_SEG_HEAD.NUM_LAYERS,
            "num_heads": cfg.MODEL.SEM_SEG_HEAD.NUM_HEADS,
            "hidden_dims": cfg.MODEL.SEM_SEG_HEAD.HIDDEN_DIMS,
            "pooling_sizes": cfg.MODEL.SEM_SEG_HEAD.POOLING_SIZES,
            "feature_resolution": cfg.MODEL.SEM_SEG_HEAD.FEATURE_RESOLUTION,
            "window_sizes": cfg.MODEL.SEM_SEG_HEAD.WINDOW_SIZES,
            "attention_type": cfg.MODEL.SEM_SEG_HEAD.ATTENTION_TYPE,

            'query_num_layers': cfg.MODEL.DIFFUSION.QUERY_ADA.NUM_LAYERS,
            'query_embed_dims': cfg.MODEL.DIFFUSION.QUERY_ADA.EMBED_DIMS,
            'query_patch_sizes': cfg.MODEL.DIFFUSION.QUERY_ADA.PATCH_SIZES,

            "DiffusionTrainer": DiffusionTrainer_(cfg),
            "transformer_predictor": CATSegPredictor(
                cfg,
            ),
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def consistloss(self, features_clip_list, query_s):
        loss_dict = {}
        for i in range(len(features_clip_list)-1):
            clip_list = []
            for ii in range(len(features_clip_list[0])):
                clip_list.append(torch.cosine_similarity(features_clip_list[i][ii].detach().clone(),
                                                     features_clip_list[i+1][ii].detach().clone()))
            query_detail = self.query_detail(query_s)
            clip_list_detail = []
            for ii in range(len(features_clip_list[0])):
                f0 = torch.einsum('cd, bdhw -> bchw', query_detail, self.clip_detail(features_clip_list[0][ii]))
                f1 = torch.einsum('cd, bdhw -> bchw', query_detail, self.clip_detail(features_clip_list[1][ii]))
                clip_list_detail.append(torch.cosine_similarity(f0, f1))
            for ii in range(len(clip_list)):
                loss_ = self.loss_smooth(clip_list_detail[ii], clip_list[ii]) * 10
                loss_dict.update({'loss_cor_' + str(i) + str(ii): loss_})

        return loss_dict

    def forward(self, batched_inputs):

        images = [x["image"].to(self.device) for x in batched_inputs]

        if self.training:
            denoise_feature, denoise_latent_list = self.diffopenseg.feature_extraction(torch.stack(images, dim=0), [50, 100, 200], self.object_query)
            for i in range(len(denoise_feature)):
                self.object_query.forward(denoise_feature[i],i)

            features_ = self.diffopenseg.layers
            self.diffopenseg.layers = []
            features_.reverse()
            denoise_img_list = []
            for latent_ in denoise_latent_list:
                denoise_img = self.diffopenseg.diff.latent_to_img_tensor(latent_.detach().clone())
                denoise_img_list.append(denoise_img)

            features_clip_list = []
            token_list = []
            clip_images = [(x - self.clip_pixel_mean) / self.clip_pixel_std for x in denoise_img_list[0]]
            clip_images = ImageList.from_tensors(clip_images, self.size_divisibility)

            self.layers = []

            clip_images_resized = F.interpolate(clip_images.tensor, size=self.clip_resolution, mode='bilinear',
                                                align_corners=False, )
            clip_features, token = self.predictor.clip_model.encode_image(clip_images_resized,
                                                                          self.object_query.return_auto(),
                                                                          dense=True)
            token_list.append(token)
            image_features = clip_features[:, 1:, :]
            res1 = rearrange(self.layers[0][1:, :, :], "(H W) B C -> B C H W", H=24)
            res2 = rearrange(self.layers[1][1:, :, :], "(H W) B C -> B C H W", H=24)
            res3 = rearrange(self.layers[2][1:, :, :], "(H W) B C -> B C H W", H=24)
            res4 = rearrange(image_features, "B (H W) C -> B C H W", H=24)
            clip_features = [res1, res2, res3, res4]
            ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
            for i in range(len(clip_features)):
                clip_features[i] = ops[i](clip_features[i])
            features_clip_list.append(clip_features)

            for ii in denoise_img_list[1:]:
                clip_images = [(x - self.clip_pixel_mean) / self.clip_pixel_std for x in ii]
                clip_images = ImageList.from_tensors(clip_images, self.size_divisibility)

                self.layers = []

                clip_images_resized = F.interpolate(clip_images.tensor, size=self.clip_resolution, mode='bilinear',
                                                    align_corners=False, )
                clip_features, token = self.predictor.clip_model.encode_image(clip_images_resized,
                                                                              dense=True)
                token_list.append(token)
                image_features = clip_features[:, 1:, :]
                res1 = rearrange(self.layers[0][1:, :, :], "(H W) B C -> B C H W", H=24)
                res2 = rearrange(self.layers[1][1:, :, :], "(H W) B C -> B C H W", H=24)
                res3 = rearrange(self.layers[2][1:, :, :], "(H W) B C -> B C H W", H=24)
                res4 = rearrange(image_features, "B (H W) C -> B C H W", H=24)
                clip_features = [res1, res2, res3, res4]
                ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
                for i in range(len(clip_features)):
                    clip_features[i] = ops[i](clip_features[i])
                features_clip_list.append(clip_features)

            loss = self.consistloss(features_clip_list, self.object_query.return_auto())

            text = self.class_texts
            text = self.predictor.get_text_embeds(text, self.prompt_templates, self.predictor.clip_model).unsqueeze(
                0).type(torch.float32)
            text = text.repeat(len(images), 1, 1, 1)
            tokens = sum([f2 for f2 in token_list[0]])
            text = self.text_query_attn(text.squeeze(2), tokens).unsqueeze(2)
            x_clip = (features_clip_list[0], tokens, text)
            targets = torch.stack([x["sem_seg"].to(self.device) for x in batched_inputs], dim=0)
            loss_clip_ = self.seg_head.loss(x_clip, targets)
            loss_clip = dict()
            for key, value in loss_clip_.items():
                loss_clip[f'clip_' + key] = value
            loss.update(loss_clip)
            return loss

        else:
            torch.cuda.empty_cache()

            clip_images = [(x - self.clip_pixel_mean) / self.clip_pixel_std for x in images]
            clip_images = ImageList.from_tensors(clip_images, self.size_divisibility)

            self.layers = []

            clip_images_resized = F.interpolate(clip_images.tensor, size=self.clip_resolution, mode='bilinear',
                                                align_corners=False, )


            clip_features, token = self.predictor.clip_model.encode_image(clip_images_resized, self.object_query.return_auto(),
                                                                          dense=True)
            image_features = clip_features[:, 1:, :]
            res1 = rearrange(self.layers[0][1:, :, :], "(H W) B C -> B C H W", H=24)
            res2 = rearrange(self.layers[1][1:, :, :], "(H W) B C -> B C H W", H=24)
            res3 = rearrange(self.layers[2][1:, :, :], "(H W) B C -> B C H W", H=24)
            res4 = rearrange(image_features, "B (H W) C -> B C H W", H=24)
            clip_features = [res1, res2, res3, res4]
            ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
            for i in range(len(clip_features)):
                clip_features[i] = ops[i](clip_features[i])

            text = self.test_class_texts
            text = self.predictor.get_text_embeds(text, self.prompt_templates, self.predictor.clip_model).unsqueeze(
                0).type(torch.float32)
            text = text.repeat(len(images), 1, 1, 1)
            tokens = sum([f2 for f2 in token])
            text = self.text_query_attn(text.squeeze(2), tokens).unsqueeze(2)
            x_clip = (clip_features, tokens, text)
            pred = self.seg_head.predict(x_clip, images.size())
            processed_results = [{'sem_seg': pred.squeeze().float()}]
            return processed_results

    @torch.no_grad()
    def inference_sliding_window(self, batched_inputs, kernel=512, overlap=0.75, out_res=[640, 640]):
        images = [x["image"].to(self.device, dtype=torch.float32) for x in batched_inputs]
        stride = int(kernel * (1 - overlap))
        unfold = nn.Unfold(kernel_size=kernel, stride=stride)
        fold = nn.Fold(out_res, kernel_size=kernel, stride=stride)

        image = F.interpolate(images[0].unsqueeze(0), size=out_res, mode='bilinear', align_corners=False).squeeze()
        image = rearrange(unfold(image), "(C H W) L-> L C H W", C=3, H=kernel)
        global_image = F.interpolate(images[0].unsqueeze(0), size=(kernel, kernel), mode='bilinear',
                                     align_corners=False)
        image = torch.cat((image, global_image), dim=0)
        batch_img_metas = []
        for xx in range(image.shape[0]):
            batch_img_metas.append({'img_shape': torch.Size([512, 512])})

        clip_images = [(x - self.clip_pixel_mean) / self.clip_pixel_std for x in image]
        clip_images = ImageList.from_tensors(clip_images, self.size_divisibility)

        self.layers = []

        clip_images_resized = F.interpolate(clip_images.tensor, size=self.clip_resolution, mode='bilinear',
                                            align_corners=False, )

        clip_features, token = self.predictor.clip_model.encode_image(clip_images_resized,
                                                                      self.object_query.return_auto(),
                                                                      dense=True)
        image_features = clip_features[:, 1:, :]
        res1 = rearrange(self.layers[0][1:, :, :], "(H W) B C -> B C H W", H=24)
        res2 = rearrange(self.layers[1][1:, :, :], "(H W) B C -> B C H W", H=24)
        res3 = rearrange(self.layers[2][1:, :, :], "(H W) B C -> B C H W", H=24)
        res4 = rearrange(image_features, "B (H W) C -> B C H W", H=24)
        clip_features = [res1, res2, res3, res4]
        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for i in range(len(clip_features)):
            clip_features[i] = ops[i](clip_features[i])

        text = self.test_class_texts
        text = self.predictor.get_text_embeds(text, self.prompt_templates, self.predictor.clip_model).unsqueeze(
            0).type(torch.float32)
        text = text.repeat(len(images), 1, 1, 1)
        tokens = sum([f2 for f2 in token])
        text = self.text_query_attn(text.squeeze(2), tokens).unsqueeze(2)
        x_clip = (clip_features, tokens, text)
        pred = self.seg_head.predict(x_clip, images.size())

        global_output = pred[-1:]
        global_output = F.interpolate(global_output, size=out_res, mode='bilinear', align_corners=False, )
        outputs = pred[:-1]
        outputs = fold(outputs.flatten(1).T) / fold(unfold(torch.ones([1] + out_res, device=self.device)))
        outputs = (outputs + global_output) / 2.

        height = batched_inputs[0].get("height", out_res[0])
        width = batched_inputs[0].get("width", out_res[1])
        output = sem_seg_postprocess(outputs[0], out_res, height, width)
        return [{'sem_seg': output}]

    def get_text_embeds(self, classnames, templates, prompt=None):

        if self.cache is not None and not self.training:
            return self.cache

        tokens = []
        for classname in classnames:
            if ', ' in classname:
                classname_splits = classname.split(', ')
                texts = [template.format(classname_splits[0]) for template in templates]
            else:
                texts = [template.format(classname) for template in templates]
            texts = self.diffopenseg.diff.tokenizer(
                texts,
                padding="max_length",
                max_length=self.diffopenseg.diff.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
                return_overflowing_tokens=True,
            ).input_ids.to(self.device)
            tokens.append(texts)
        tokens = torch.stack(tokens, dim=0).squeeze(1)
        if prompt is None:
            self.diffopenseg.diff.tokens = tokens

        class_embeddings = self.diffopenseg.diff.text_encoder(tokens).pooler_output

        class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

        class_embeddings = class_embeddings.unsqueeze(1)

        if not self.training:
            self.cache = class_embeddings

        return class_embeddings
