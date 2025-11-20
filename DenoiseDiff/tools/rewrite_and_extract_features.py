import torch
import torch.nn.functional as F
from diffusers.models.transformer_2d import Transformer2DModelOutput


def init_block_func(
        unet,
        mode,
        save_hidden=False,
        use_hidden=False,
        reset=True,
        save_timestep=[],
        idxs=[(1, 0)],
        flag_layer='resnet',
        adapter_=None,
):
    def renet_new_forward(self, input_tensor, temb):
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            input_tensor = self.upsample(input_tensor)
            hidden_states = self.upsample(hidden_states)
        elif self.downsample is not None:
            input_tensor = self.downsample(input_tensor)
            hidden_states = self.downsample(hidden_states)

        hidden_states = self.conv1(hidden_states)

        if temb is not None:
            temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]
            hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        if save_hidden:
            if save_timestep is None or self.timestep in save_timestep:
                self.feats[self.timestep] = hidden_states
        elif use_hidden:
            hidden_states = self.feats[self.timestep]
        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        return output_tensor

    def transformer_new_forward(
            self,
            hidden_states,
            encoder_hidden_states=None,
            timestep=None,
            class_labels=None,
            cross_attention_kwargs=None,
            return_dict: bool = True,
    ):

        if self.is_input_continuous:
            batch, _, height, width = hidden_states.shape
            residual = hidden_states

            hidden_states = self.norm(hidden_states)
            if not self.use_linear_projection:
                hidden_states = self.proj_in(hidden_states)
                inner_dim = hidden_states.shape[1]
                hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
            else:
                inner_dim = hidden_states.shape[1]
                hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
                hidden_states = self.proj_in(hidden_states)
        elif self.is_input_vectorized:
            hidden_states = self.latent_image_embedding(hidden_states)
        elif self.is_input_patches:
            hidden_states = self.pos_embed(hidden_states)

        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=class_labels,
            )

        if self.is_input_continuous:
            if not self.use_linear_projection:
                hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
                hidden_states = self.proj_out(hidden_states)
            else:
                hidden_states = self.proj_out(hidden_states)
                hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

            output = hidden_states + residual
        elif self.is_input_vectorized:
            hidden_states = self.norm_out(hidden_states)
            logits = self.out(hidden_states)
            logits = logits.permute(0, 2, 1)

            output = F.log_softmax(logits.double(), dim=1).float()
        elif self.is_input_patches:
            conditioning = self.transformer_blocks[0].norm1.emb(
                timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
            shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
            hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
            hidden_states = self.proj_out_2(hidden_states)

            height = width = int(hidden_states.shape[1] ** 0.5)
            hidden_states = hidden_states.reshape(
                shape=(-1, height, width, self.patch_size, self.patch_size, self.out_channels)
            )
            hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
            output = hidden_states.reshape(
                shape=(-1, self.out_channels, height * self.patch_size, width * self.patch_size)
            )

        if save_hidden:
            if save_timestep is None or self.timestep in save_timestep:
                self.feats[self.timestep] = hidden_states

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    def attn_new_forward(
            self,
            hidden_states,
            encoder_hidden_states=None,
            timestep=None,
            attention_mask=None,
            cross_attention_kwargs=None,
            class_labels=None,
    ):
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        else:
            norm_hidden_states = self.norm1(hidden_states)

        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = attn_output + hidden_states

        if self.attn2 is not None:
            norm_hidden_states = (
                self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
            )

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        norm_hidden_states = self.norm3(hidden_states)

        if self.use_ada_layer_norm_zero:
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)

        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = ff_output + hidden_states

        if self.mode_ == 'down_blocks':
            adapter_[self.index_[0][0]].forward(
                hidden_states,
                self.index_[0][1],
                batch_first=True,
                has_cls_token=False, )
        elif self.mode_ == 'up_blocks':
            adapter_[self.index_[0][0]+3].forward(
                hidden_states,
                self.index_[0][1],
                batch_first=True,
                has_cls_token=False, )
        elif self.mode_ == 'mid_block':
            adapter_[self.index_[0][0] + 3].forward(
                hidden_states,
                self.index_[0][1],
                batch_first=True,
                has_cls_token=False, )

        return hidden_states

    if 'resnet' == flag_layer:
        layers = collect_layers(unet, mode, idxs, flag_layer='resnet')
        for module in layers:
            module.forward = renet_new_forward.__get__(module, type(module))
            if reset:
                module.feats = {}
                module.timestep = None
    if 'attention' == flag_layer:
        layers = collect_layers(unet, mode, idxs, flag_layer='attention')
        for module in layers:
            module.forward = transformer_new_forward.__get__(module, type(module))
            if reset:
                module.feats = {}
                module.timestep = None
    if 'attention_module_query' == flag_layer:
        idxs = [[(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)],
                [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)], [(0, 0)]]
        mode = ['down_blocks', 'up_blocks', 'mid_block']
        for ii in range(len(idxs)):
            layers = collect_layers_attn(unet, mode[ii], idxs[ii], flag_layer='attention')
            for index_t, module in enumerate(layers):
                module.forward = attn_new_forward.__get__(module, type(module))
                module.index_ = [idxs[ii][index_t]]
                module.mode_ = mode[ii]


def collect_layers_attn(unet, mode, idxs=None, flag_layer='resnet'):
    layers = []
    if flag_layer == 'attention':
        if mode == 'up_blocks':
            for i, up_block in enumerate(unet.up_blocks):
                if hasattr(up_block, 'attentions'):
                    for j, module in enumerate(up_block.attentions):
                        if idxs is None or (i, j) in idxs:
                            layers.append(module.transformer_blocks[0])
        elif mode == 'down_blocks':
            for i, down_block in enumerate(unet.down_blocks):
                if hasattr(down_block, 'attentions'):
                    for j, module in enumerate(down_block.attentions):
                        if idxs is None or (i, j) in idxs:
                            layers.append(module.transformer_blocks[0])
        elif mode == 'mid_block':
            for j, module in enumerate(unet.mid_block.attentions):
                if idxs is None or (0, j) in idxs:
                    layers.append(module.transformer_blocks[0])
    return layers


def collect_layers(unet, mode, idxs=None, flag_layer='resnet'):
    layers = []
    if flag_layer == 'resnet':
        if mode == 'up_blocks':
            for i, up_block in enumerate(unet.up_blocks):
                for j, module in enumerate(up_block.resnets):
                    if idxs is None or (i, j) in idxs:
                        layers.append(module)
        elif mode == 'down_blocks':
            for i, down_block in enumerate(unet.down_blocks):
                for j, module in enumerate(down_block.resnets):
                    if idxs is None or (i, j) in idxs:
                        layers.append(module)
        elif mode == 'mid_block':
            for i, mid_block in enumerate(unet.mid_block):
                for j, module in enumerate(mid_block.resnets):
                    if idxs is None or (i, j) in idxs:
                        layers.append(module)
    if flag_layer == 'attention':
        if mode == 'up_blocks':
            for i, up_block in enumerate(unet.up_blocks):
                if hasattr(up_block, 'attentions'):
                    for j, module in enumerate(up_block.attentions):
                        if idxs is None or (i, j) in idxs:
                            layers.append(module)
        elif mode == 'down_blocks':
            for i, down_block in enumerate(unet.down_blocks):
                if hasattr(down_block, 'attentions'):
                    for j, module in enumerate(down_block.attentions):
                        if idxs is None or (i, j) in idxs:
                            layers.append(module)
        elif mode == 'mid_block':
            for i, mid_block in enumerate(unet.mid_block):
                for j, module in enumerate(mid_block.attentions):
                    if idxs is None or (i, j) in idxs:
                        layers.append(module)
    return layers


def collect_feats(unet, mode, idxs, flag_layer='resnet'):
    feats = []
    layers = collect_layers(unet, mode, idxs, flag_layer)
    for module in layers:
        feats.append(module.feats)
        module.feats = {}
        module.timestep = None
    return feats


def set_timestep(unet, layer_indexes, timestep=None):
    layers = collect_layers(unet, 'up_blocks', layer_indexes, 'resnet')
    for module in layers:
        module_name = type(module).__name__
        module.timestep = timestep

