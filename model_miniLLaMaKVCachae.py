import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# KV Cache 容器
@dataclass
class KVCache:
    k: torch.Tensor
    v: torch.Tensor


@dataclass
class Output:
    tensor: torch.Tensor
    kvcache: KVCache

@dataclass
class Transformeroutput:
    output: torch.Tensor
    kvcachelist: list[KVCache]

@dataclass
class ModelConfig:
    vocab_size: int = 50257
    block_size: int = 128

    num_layers: int = 6
    d_model: int = 384
    d_ffn: int = 1024
    num_heads: int = 4

    rms_norm_eps: float = 1e-6

    def __post_init__(self):
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )

        self.d_head = self.d_model // self.num_heads

        if self.d_head % 2 != 0:
            raise ValueError(
                f"d_head ({self.d_head}) must be even for RoPE"
            )


# y = f(x), x 为 q 或者 k，输出为加上了位置编码后的 q' 或 k'
# Prefill 阶段, position 为长度为 seq_l 的数组，Decode 阶段为长为 1 的数组，内容为 token id
def apply_rope(x: torch.Tensor, position: torch.tensor) -> torch.Tensor:
    B, seq_len, d_head = x.shape
    assert d_head % 2 == 0

    # position = torch.arange(seq_len, device=x.device)

    freq = 1.0 / (10000 ** (torch.arange(0, d_head, 2, device=x.device) / d_head))

    theta = position[:, None] * freq[None, :]

    cos = torch.cos(theta)
    sin = torch.sin(theta)

    cos = cos.unsqueeze(0)
    sin = sin.unsqueeze(0)

    x_even = x[:, :, 0::2]
    x_odd = x[:, :, 1::2]

    x_even_new = x_even * cos - x_odd * sin
    x_odd_new = x_even * sin + x_odd * cos

    x_rope = torch.stack([x_even_new, x_odd_new], dim=-1).flatten(-2)

    return x_rope


class MaskMultiHeadAttentionRoPE(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.d_head = config.d_head

        self.wq = nn.Parameter(torch.randn(self.d_model, self.d_model))
        self.wk = nn.Parameter(torch.randn(self.d_model, self.d_model))
        self.wv = nn.Parameter(torch.randn(self.d_model, self.d_model))

        self.wo = nn.Parameter(torch.randn(self.d_model, self.d_model))

        nn.init.xavier_uniform_(self.wq)
        nn.init.xavier_uniform_(self.wk)
        nn.init.xavier_uniform_(self.wv)
        nn.init.xavier_uniform_(self.wo)

    # Prefill 阶段无 KVCache 输入，Decode 阶段有, mmha 无需分辨，调用者注意分辨即可
    def forward(self, x, kvcache=None, position=None) -> Output:
        B, L, _ = x.shape

        q = x @ self.wq
        k = x @ self.wk
        v = x @ self.wv

        q = q.view(B, L, self.num_heads, self.d_head)
        k = k.view(B, L, self.num_heads, self.d_head)
        v = v.view(B, L, self.num_heads, self.d_head)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = q.reshape(B * self.num_heads, L, self.d_head)
        k = k.reshape(B * self.num_heads, L, self.d_head)

        q = apply_rope(q, position)
        k = apply_rope(k, position)

        q = q.view(B, self.num_heads, L, self.d_head)
        k = k.view(B, self.num_heads, L, self.d_head)

        if kvcache is not None:
            k = torch.cat([kvcache.k, k], dim=2)
            v = torch.cat([kvcache.v, v], dim=2)

        attention_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # Decode 阶段不需要 mask
        if kvcache is None and L > 1:
            mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), diagonal=1)
            attention_scores = attention_scores.masked_fill(mask, float("-inf"))

        attention_probs = torch.softmax(attention_scores, dim=-1)

        attention = attention_probs @ v
        attention = attention.transpose(1, 2)

        attention = attention.contiguous().view(B, L, self.d_model)
        attention = attention @ self.wo

        output = Output(tensor=attention, kvcache=KVCache(k=k, v=v))

        return output


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ffn):
        super().__init__()
        self.d_model = d_model
        self.d_ffn = d_ffn

        self.w1 = nn.Parameter(
            torch.randn(d_model, d_ffn)
        )

        self.w2 = nn.Parameter(
            torch.randn(d_ffn, d_model)
        )

        self.w3 = nn.Parameter(
            torch.randn(d_model, d_ffn)
        )

        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)
        nn.init.xavier_uniform_(self.w3)

    def forward(self, x):
        x1 = F.silu(x @ self.w1)
        y = x @ self.w3
        z = x1 * y
        x = z @ self.w2
        return x


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.eps = eps

        self.gamma = nn.Parameter(
            torch.ones(d_model)
        )

    def forward(self, x):
        rms = torch.sqrt(
            (x ** 2).mean(
                dim=-1,
                keepdim=True
            )
            + self.eps
        )
        x = (x / rms) * self.gamma
        return x


class LMHead(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        self.wlm = nn.Parameter(
            torch.randn(d_model, vocab_size)
        )

        nn.init.xavier_uniform_(self.wlm)

    def forward(self, x):
        x = x @ self.wlm
        return x


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.rmsnorm1 = RMSNorm(config.d_model,config.rms_norm_eps)
        self.rmsnorm2 = RMSNorm(config.d_model,config.rms_norm_eps)
        self.mmha = MaskMultiHeadAttentionRoPE(config)
        self.swiglu = SwiGLU(config.d_model,config.d_ffn)

    def forward(self, x, kvcache=None, position=None):

        mmha_output = self.mmha(self.rmsnorm1(x), kvcache, position)

        kvcache = mmha_output.kvcache
        x = x + mmha_output.tensor

        x = x + self.swiglu(self.rmsnorm2(x))
        output = Output(tensor=x, kvcache=kvcache)
        return output


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        self.embedding = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.d_model)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_layers)])
        self.rmsnorm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.lm_head = LMHead(config.d_model, config.vocab_size)

    def forward(self, x, kvcachelist=None, position=None):
        x = self.embedding(x)
        new_kvcachelist = []
        for i, layer in enumerate(self.layers):
            if kvcachelist is None:
                layer_output = layer(x, None, position)
            else:
                layer_output = layer(x, kvcachelist[i], position)
            new_kvcachelist.append(layer_output.kvcache)
            x = layer_output.tensor

        x = self.rmsnorm(x)
        x = self.lm_head(x)

        output = Transformeroutput(output=x, kvcachelist=new_kvcachelist)
        return output

