import torch
from torch import nn
import torch.nn.functional as F
from torch.func import functional_call, jvp
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.masking_utils import create_causal_mask

from jvp_flash_attention.jvp_attention import JVPAttn


ATOL = 1e-5


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


def max_diff(a, b):
    return torch.max(torch.abs(a - b)).item()


def summarize_stage(name, eager_primal, compiled_primal, eager_tangent, compiled_tangent):
    primal = max_diff(eager_primal, compiled_primal)
    tangent = max_diff(eager_tangent, compiled_tangent)
    print(f"{name}:")
    print(f"  max|eager_primal - compiled_primal| = {primal:.6f}")
    print(f"  max|eager_jvp - compiled_jvp| = {tangent:.6f}")
    return {"name": name, "primal": primal, "tangent": tangent}


def compare_input_jvp(name, func, primals, tangents):
    eager_primal, eager_tangent = jvp(func, primals, tangents)

    @torch.compile(fullgraph=True)
    def compiled_stage(*stage_tangents):
        return jvp(func, primals, stage_tangents)

    compiled_primal, compiled_tangent = compiled_stage(*tangents)
    return summarize_stage(name, eager_primal, compiled_primal, eager_tangent, compiled_tangent)


def compare_param_jvp(name, func, params, tangents):
    eager_primal, eager_tangent = jvp(func, (params,), (tangents,))

    @torch.compile(fullgraph=True)
    def compiled_stage(stage_tangents):
        return jvp(func, (params,), (stage_tangents,))

    compiled_primal, compiled_tangent = compiled_stage(tangents)
    return summarize_stage(name, eager_primal, compiled_primal, eager_tangent, compiled_tangent)


def run_stage(name, fn):
    print()
    try:
        return fn()
    except Exception as exc:
        print(f"{name}: FAILED")
        print(f"  {type(exc).__name__}: {exc}")
        return {"name": name, "primal": float("inf"), "tangent": float("inf")}


if __name__ == "__main__":
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    F.scaled_dot_product_attention = jvp_attention_wrapper

    vocab_size = 100
    config = GPT2Config(
        vocab_size=vocab_size,
        n_embd=64,
        n_layer=1,
        n_head=2,
        n_positions=32,
        use_cache=False,
    )
    model = GPTWrapperModel(GPT2LMHeadModel(config).to(device))
    model.eval()

    transformer = model.model.transformer
    block = transformer.h[0]
    attn = block.attn

    batch_size, seq_len = 1, 32
    input_batch = {
        "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len), device=device),
        "attention_mask": torch.ones((batch_size, seq_len), device=device),
        "position_ids": torch.arange(seq_len, dtype=torch.long, device=device).view(
            batch_size, seq_len
        ),
    }

    cache_position = torch.arange(seq_len, device=device, dtype=torch.long)
    attention_mask = input_batch["attention_mask"].view(batch_size, -1)

    with torch.no_grad():
        inputs_embeds = transformer.wte(input_batch["input_ids"])
        position_embeds = transformer.wpe(input_batch["position_ids"])
        block_hidden_states = transformer.drop(inputs_embeds + position_embeds)
        causal_mask = create_causal_mask(
            config=transformer.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=input_batch["position_ids"],
        )
        attn_hidden_states = block.ln_1(block_hidden_states)

    print(
        "wte/lm_head tied:",
        model.model.transformer.wte.weight.data_ptr() == model.model.lm_head.weight.data_ptr(),
    )

    results = []

    q = torch.randn(1, 2, 32, 32, device=device)
    k = torch.randn(1, 2, 32, 32, device=device)
    v = torch.randn(1, 2, 32, 32, device=device)
    q_t = torch.randn_like(q)
    k_t = torch.randn_like(k)
    v_t = torch.randn_like(v)

    def direct_attn(q_, k_, v_):
        return JVPAttn.fwd_dual(q_, k_, v_, causal=True)

    results.append(
        run_stage(
            "direct_jvp_attention",
            lambda: compare_input_jvp("direct_jvp_attention", direct_attn, (q, k, v), (q_t, k_t, v_t)),
        )
    )

    attn_params = dict(attn.named_parameters())
    attn_tangents = {k_: torch.randn_like(v_) for k_, v_ in attn_params.items()}

    def attention_stage(p):
        return functional_call(
            attn,
            p,
            args=(attn_hidden_states,),
            kwargs={
                "past_key_values": None,
                "cache_position": cache_position,
                "attention_mask": causal_mask,
                "use_cache": False,
                "output_attentions": False,
                "position_ids": input_batch["position_ids"],
            },
        )[0]

    results.append(
        run_stage(
            "gpt2_attention_module",
            lambda: compare_param_jvp(
                "gpt2_attention_module", attention_stage, attn_params, attn_tangents
            ),
        )
    )

    block_params = dict(block.named_parameters())
    block_tangents = {k_: torch.randn_like(v_) for k_, v_ in block_params.items()}

    def block_stage(p):
        return functional_call(
            block,
            p,
            args=(block_hidden_states,),
            kwargs={
                "past_key_values": None,
                "cache_position": cache_position,
                "attention_mask": causal_mask,
                "use_cache": False,
                "position_ids": input_batch["position_ids"],
            },
        )

    results.append(
        run_stage(
            "gpt2_block",
            lambda: compare_param_jvp("gpt2_block", block_stage, block_params, block_tangents),
        )
    )

    transformer_params = dict(transformer.named_parameters())
    transformer_tangents = {
        k_: torch.randn_like(v_) for k_, v_ in transformer_params.items()
    }

    def transformer_stage(p):
        outputs = functional_call(
            transformer,
            p,
            args=(),
            kwargs={
                "input_ids": input_batch["input_ids"],
                "attention_mask": input_batch["attention_mask"],
                "position_ids": input_batch["position_ids"],
                "use_cache": False,
                "return_dict": True,
            },
        )
        return outputs.last_hidden_state

    results.append(
        run_stage(
            "gpt2_transformer",
            lambda: compare_param_jvp(
                "gpt2_transformer",
                transformer_stage,
                transformer_params,
                transformer_tangents,
            ),
        )
    )

    full_params = dict(model.named_parameters())
    full_tangents = {k_: torch.randn_like(v_) for k_, v_ in full_params.items()}

    def full_stage(p):
        return functional_call(model, p, input_batch)

    results.append(
        run_stage(
            "gpt2_lm_wrapper",
            lambda: compare_param_jvp("gpt2_lm_wrapper", full_stage, full_params, full_tangents),
        )
    )

    print("\nFirst stage exceeding atol=1e-5:")
    first_bad = next(
        (r for r in results if r["primal"] > ATOL or r["tangent"] > ATOL),
        None,
    )
    if first_bad is None:
        print("  none")
    else:
        print(
            f"  {first_bad['name']} "
            f"(primal={first_bad['primal']:.6f}, tangent={first_bad['tangent']:.6f})"
        )
