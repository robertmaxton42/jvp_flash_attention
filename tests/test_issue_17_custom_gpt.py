import torch
from torch import nn
import torch.nn.functional as F
from torch.func import functional_call, jvp
from jvp_flash_attention.jvp_attention import JVPAttn


def jvp_attention_wrapper(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
    **kwargs,
):
    return JVPAttn.fwd_dual(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=0.0,
        causal=is_causal,
        sm_scale=scale,
        verify_attn_mask=False,
        **kwargs,
    )


class MinimalGPT2Attention(nn.Module):
    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_len, n_embd = hidden_states.shape
        query, key, value = self.c_attn(hidden_states).chunk(3, dim=-1)

        query = query.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(
            1, 2
        )
        key = key.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(
            1, 2
        )

        attn_weights = query @ key.transpose(-1, -2)
        attn_weights = attn_weights / (self.head_dim**0.5)

        causal_bias = torch.full(
            (seq_len, seq_len),
            torch.finfo(attn_weights.dtype).min,
            dtype=attn_weights.dtype,
            device=attn_weights.device,
        )
        causal_bias = torch.triu(causal_bias, diagonal=1)
        attn_weights = attn_weights + causal_bias[None, None, :, :]

        padding_bias = 1.0 - attention_mask[:, None, None, :].to(attn_weights.dtype)
        padding_bias = padding_bias * torch.finfo(attn_weights.dtype).min
        attn_weights = attn_weights + padding_bias

        attn_probs = torch.softmax(attn_weights, dim=-1)
        attn_output = attn_probs @ value
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, n_embd)
        return self.c_proj(attn_output)


class MinimalGPT2Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = MinimalGPT2Attention(n_embd=n_embd, n_head=n_head)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.ln_1(hidden_states), attention_mask
        )
        hidden_states = hidden_states + self.mlp(self.ln_2(hidden_states))
        return hidden_states


class MinimalGPT2LMHeadModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 100,
        n_embd: int = 64,
        n_layer: int = 1,
        n_head: int = 2,
        n_positions: int = 32,
    ):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(n_positions, n_embd)
        self.h = nn.ModuleList(
            [MinimalGPT2Block(n_embd=n_embd, n_head=n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.wte(input_ids) + self.wpe(position_ids)
        for block in self.h:
            hidden_states = block(hidden_states, attention_mask)
        hidden_states = self.ln_f(hidden_states)
        return self.lm_head(hidden_states)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    F.scaled_dot_product_attention = jvp_attention_wrapper

    torch.manual_seed(0)

    model = MinimalGPT2LMHeadModel().to(device)
    model.eval()

    batch_size, seq_len, vocab_size = 1, 32, 100
    input_ids = {
        "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len), device=device),
        "attention_mask": torch.rand((batch_size, seq_len), device=device).round().bool(),
        "position_ids": torch.arange(seq_len, dtype=torch.long, device=device).view(
            batch_size, seq_len
        ),
    }

    params = dict(model.named_parameters())
    tangents = {k: torch.randn_like(v) for k, v in params.items()}

    def func_model(p):
        return functional_call(model, p, (), input_ids)

    out_eager = jvp(func_model, (params,), (tangents,))[1]
    print("Eager JVP success:", out_eager.shape)

    @torch.compile(fullgraph=True)
    def compiled_jvp(t):
        return jvp(func_model, (params,), (t,))[1]

    out_compiled = compiled_jvp(tangents)
    print("Compiled JVP success:", out_compiled.shape)

    max_diff = torch.max(torch.abs(out_eager - out_compiled))
    assert torch.allclose(out_eager, out_compiled, atol=5e-5), (
        "Eager and Compiled JVP outputs differ by "
        f"{max_diff} for the minimal GPT-2 reproduction!"
    )
    print("Eager and Compiled JVP outputs are close!")
