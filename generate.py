import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast

from ModelStructures.model_miniLLaMa import ModelConfig, MiniLLaMA

def load_model(checkpoint_path,device,):
    checkpoint = torch.load(checkpoint_path, map_location=device,)
    model_config_dict = checkpoint["model_config"]
    model_config = ModelConfig(**model_config_dict)

    model = MiniLLaMA(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Checkpoint loss: {checkpoint.get('loss', 'unknown')}")

    return model, model_config


def apply_top_k(logits,top_k,):
    if top_k <= 0:
        return logits

    vocab_size = logits.size(-1)
    top_k = min(top_k,vocab_size,)

    values, _ = torch.topk(logits, top_k, dim=-1,)

    # 第 k 大的 logit
    min_values = values[:, -1].unsqueeze(-1)
    logits = logits.masked_fill(logits < min_values,float("-inf"),)

    return logits


def apply_top_p(logits, top_p):
    if top_p >= 1.0:
        return logits
    if top_p <= 0.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = (cumulative_probs > top_p)

    # 保留第一个超过 top_p 的 token
    sorted_indices_to_remove[:, 1:] = (sorted_indices_to_remove[:, :-1].clone())
    sorted_indices_to_remove[:, 0] = False

    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
    indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)

    logits = logits.masked_fill(indices_to_remove, float("-inf"),)

    return logits


def sample_next_token(
    logits,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    do_sample=True,
):
    # 不采样，直接返回概率最大者
    if not do_sample:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    logits = logits / temperature
    logits = apply_top_k(logits, top_k)
    logits = apply_top_p(logits, top_p)
    probs = F.softmax(logits, dim=-1)

    # 随机采样
    next_token = torch.multinomial(probs, num_samples=1)

    return next_token


# 不使用 KV Cache 的生成
@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
    device,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    for _ in range(max_new_tokens):

        # 每次都重新计算完整上下文
        input_context = input_ids[:, -model.config.block_size:]

        position = torch.arange(input_context.shape[1], device=input_ids.device)

        # 不使用 KV Cache
        model_output = model(input_context, None, position)

        logits = model_output.output

        # 取最后一个位置的 logits
        next_token_logits = logits[:, -1, :]

        # 采样 / greedy
        next_token = sample_next_token(
            logits=next_token_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
        )

        # 保存生成结果
        input_ids = torch.cat([input_ids, next_token],dim=1)

    generated_text = tokenizer.decode(input_ids[0],skip_special_tokens=True)
    return generated_text


@torch.no_grad()
def generate_kvcache(
    model,
    tokenizer,
    prompt,
    device,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
):
    # 输出为 "pt" : PyTorch Tensor ，所以会包装一个维度，相当于 [B,L] ，只不过 B=1 ，所以基本上无论训练还是推理，
    # 模型内部接受的输入都会有个 Batch，接口设计如此。
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.to(device)
    kvcachelist = None

    input_context = input_ids

    for i in range(max_new_tokens):
        # 截断上下文至模型上下文窗口大小
        # 第一版 KV Cache 暂时不考虑上下文大小问题，即 KV Cache 会无限增大（直到 max_new_tokens）
        # input_context = input_ids[:, -model.config.block_size:]


        if i == 0:
            position = torch.arange(input_context.shape[1], device=input_ids.device)
        else:
            position = torch.tensor([input_ids.shape[1] - 1], device=input_ids.device)

        # 调用模型 forward 生成 logits 分数
        model_output = model(input_context, kvcachelist, position)
        logits = model_output.output
        kvcachelist = model_output.kvcachelist


        # 取最后一行 logits，投机解码时可以在这里做手脚
        next_token_logits = logits[:, -1, :]

        # 对模型生成 logits 分数进行采样得到下个 token
        next_token = sample_next_token(
            logits=next_token_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
        )

        # 拼接 token 至已有输出
        input_ids = torch.cat([input_ids, next_token], dim=1)

        # KV Cache 了则只需要输入新的 token 就行
        input_context = next_token

    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text


# 暴露给 cli 的接口
def generate_from_checkpoint(
    checkpoint_path,
    prompt,
    device=None,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    tokenizer_name="gpt2",
):
    if device is None:device = ("cuda"if torch.cuda.is_available()else "cpu")

    device = torch.device(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)

    model, model_config = load_model(checkpoint_path=checkpoint_path, device=device)

    text = generate_kvcache(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    return text


if __name__ == "__main__":
    raise RuntimeError(
        "Please run generation through cli.py."
    )
