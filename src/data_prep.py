
def load_dataset(*args, **kwargs):
    # Keep datasets optional until real evaluation data is requested.  This is
    # important for the self-contained synthetic smoke path.
    from datasets import load_dataset as datasets_load_dataset
    return datasets_load_dataset(*args, **kwargs)

def prepare_guanaco(data_path, batch_size, cache_dir=None):
    dataset = load_dataset(data_path, split="train", cache_dir=cache_dir)
    texts = dataset['text']
    longest_texts = sorted(texts, key=len, reverse=True)[:batch_size]
    return longest_texts


def prepare_longform(data_path, batch_size, cache_dir=None):
    dataset = load_dataset(data_path, split="train", cache_dir=cache_dir)
    def format_longform(example):
        example["text"] = f"### Instruction:\n{example['input']}\n\n### Response:\n{example['output']}"
        return example

    formatted_dataset = dataset.map(format_longform)
    texts = formatted_dataset['text']
    longest_texts = sorted(texts, key=len, reverse=True)[:batch_size]
    return longest_texts

def prepare_oasst1(data_path: str, batch_size: int, cache_dir=None):
    """
    Convert OpenAssistant/oasst1 conversation trees into
    alpaca-style (instruction, response) pairs.
    """
    ds = load_dataset(data_path, split="train", cache_dir=cache_dir)
    # 建立 message_id -> text 的查表
    id2text = {ex["message_id"]: ex["text"] for ex in ds}
    pairs = []
    for ex in ds:
        if ex["role"] != "assistant":
            continue                         # 只取 Assistant 回复
        parent_id = ex["parent_id"]
        if parent_id is None:
            continue                         # 根节点，跳过
        instr = id2text.get(parent_id)
        if instr is None:
            continue                         # parent 不在同一个 split
        resp  = ex["text"]
        pairs.append(f"### Instruction:\n{instr}\n\n### Response:\n{resp}")

    # 取最长的 batch_size 条
    longest_texts = sorted(pairs, key=len, reverse=True)[:batch_size]
    return longest_texts

def data_prepare(data_path, batch_size, cache_dir=None):
    if "guanaco" in data_path:
        return prepare_guanaco(data_path, batch_size, cache_dir)
    elif "LongForm" in data_path:
        return prepare_longform(data_path, batch_size, cache_dir)
    elif "oasst1" in data_path.lower():
        return prepare_oasst1(data_path, batch_size, cache_dir)
    else:
        raise ValueError(f"Unsupported dataset: {data_path}")
