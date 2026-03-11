import math
import torch
import torch.nn.functional as F
from torch import einsum


class FluxVtonAttnProcessor:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self, store_list=None):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        # self.store_list = store_list

    def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            image_rotary_emb=None,
            ref_rotary_emb=None,
            in_mask=None,
            out_mask=None,
            height=512,
            width=384,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        emb_len = 512

        HW = hidden_states.shape[1] - emb_len if encoder_hidden_states is None else hidden_states.shape[1]
        W = int(math.sqrt(HW / (height / width)))
        H = HW // W

        dtype = hidden_states.dtype

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            if ref_rotary_emb is not None:
                key_out, key_in = key.chunk(2)
                if key.shape[2] * 2 > ref_rotary_emb[0].shape[0]:
                    hyd_key = torch.cat([key_in, key_out[:, :, emb_len:]], dim=2)
                    ref_key = torch.cat([key_out, key_in[:, :, emb_len:]], dim=2)
                else:
                    hyd_key = torch.cat([key_in, key_out], dim=2)
                hyd_key = apply_rotary_emb(hyd_key, ref_rotary_emb)
                ref_key = apply_rotary_emb(ref_key, ref_rotary_emb)
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        if ref_rotary_emb is not None:
            query_out, query_in = query.chunk(2)
            value_out, value_in = value.chunk(2)

            ori_sim = einsum("b h i d, b h j d -> b h i j", query_out, ref_key) / math.sqrt(query_out.size(-1))
            ori_sim = ori_sim.softmax(-1).to(dtype)
            ref_value = torch.cat([value_out, value_in[:, :, emb_len:]], dim=2)
            ori_output = einsum("b h i j, b h j d -> b h i d", ori_sim, ref_value)

            if key.shape[2] * 2 > ref_rotary_emb[0].shape[0]:
                hyd_v = torch.cat([value_in, value_out[:, :, emb_len:]], dim=2).to(dtype)
            else:
                hyd_v = torch.cat([value_in, value_out], dim=2).to(dtype)

            hyd_sim = einsum("b h i d, b h j d -> b h i j", query_in, hyd_key) / math.sqrt(query_in.size(-1))
            hyd_sim = hyd_sim.softmax(-1)
            cross_probs = hyd_sim[:, :, :, emb_len - query_in.shape[2]:]
            hyd_sim[:, :, :, emb_len - query_in.shape[2]:] = cross_probs
            hyd_output = einsum("b h i j, b h j d -> b h i d", hyd_sim, hyd_v)

            hidden_states = torch.cat([ori_output, hyd_output])
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, :encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1]:],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states