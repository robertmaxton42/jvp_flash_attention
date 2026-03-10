import torch
from torch import nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel
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


class GPTWrapperModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, data):
        input_ids = data["input_ids"].to(self.model.device)
        attention_mask = data["attention_mask"].to(self.model.device)
        position_ids = data["position_ids"].to(self.model.device)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=None,
        )
        return outputs.logits


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    F.scaled_dot_product_attention = jvp_attention_wrapper

    torch.manual_seed(0)

    vocab_size = 100
    config = GPT2Config(
        vocab_size=vocab_size,
        n_embd=64,
        n_layer=1,
        n_head=2,
        n_positions=32,
        use_cache=False,
    )
    config._attn_implementation = "eager" # Required to avoid causal mask issue when compiling

    model = GPTWrapperModel(GPT2LMHeadModel(config).to(device))
    model.eval()

    batch_size, seq_len = 1, 32
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
        return functional_call(model, p, input_ids)

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
        f"{max_diff} for Hugging Face's GPT-2 model!"
    )
    print("Eager and Compiled JVP outputs are close!")
