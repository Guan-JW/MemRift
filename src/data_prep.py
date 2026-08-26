
def load_dataset(*args, **kwargs):
    # Keep datasets optional until real evaluation data is requested.  This is
    # important for the self-contained synthetic smoke path.
    from datasets import load_dataset as datasets_load_dataset
    return datasets_load_dataset(*args, **kwargs)

def prepare_guanaco(data_path, batch_size, cache_dir=None, revision=None):
    dataset = load_dataset(data_path, split="train", cache_dir=cache_dir, revision=revision)
    texts = dataset['text']
    longest_texts = sorted(texts, key=len, reverse=True)[:batch_size]
    return longest_texts


def prepare_longform(data_path, batch_size, cache_dir=None, revision=None):
    dataset = load_dataset(data_path, split="train", cache_dir=cache_dir, revision=revision)
    def format_longform(example):
        example["text"] = f"### Instruction:\n{example['input']}\n\n### Response:\n{example['output']}"
        return example

    formatted_dataset = dataset.map(format_longform)
    texts = formatted_dataset['text']
    longest_texts = sorted(texts, key=len, reverse=True)[:batch_size]
    return longest_texts

def prepare_oasst1(data_path: str, batch_size: int, cache_dir=None, revision=None):
    """
    Convert OpenAssistant/oasst1 conversation trees into
    alpaca-style (instruction, response) pairs.
    """
    ds = load_dataset(data_path, split="train", cache_dir=cache_dir, revision=revision)
    # Build a message_id-to-text lookup for conversation parents.
    id2text = {ex["message_id"]: ex["text"] for ex in ds}
    pairs = []
    for ex in ds:
        if ex["role"] != "assistant":
            continue
        parent_id = ex["parent_id"]
        if parent_id is None:
            continue
        instr = id2text.get(parent_id)
        if instr is None:
            continue
        resp  = ex["text"]
        pairs.append(f"### Instruction:\n{instr}\n\n### Response:\n{resp}")

    longest_texts = sorted(pairs, key=len, reverse=True)[:batch_size]
    return longest_texts


def prepare_alpaca(data_path, batch_size, cache_dir=None, revision=None):
    dataset = load_dataset(data_path, split="train", cache_dir=cache_dir, revision=revision)
    texts = []
    for example in dataset:
        instruction = example["instruction"]
        input_text = example.get("input", "")
        prompt = instruction if not input_text else f"{instruction}\n\nInput:\n{input_text}"
        texts.append(f"### Instruction:\n{prompt}\n\n### Response:\n{example['output']}")
    return sorted(texts, key=len, reverse=True)[:batch_size]


def data_prepare(data_path, batch_size, cache_dir=None, revision=None):
    if "guanaco" in data_path:
        return prepare_guanaco(data_path, batch_size, cache_dir, revision)
    elif "LongForm" in data_path:
        return prepare_longform(data_path, batch_size, cache_dir, revision)
    elif "oasst1" in data_path.lower():
        return prepare_oasst1(data_path, batch_size, cache_dir, revision)
    elif "alpaca" in data_path.lower():
        return prepare_alpaca(data_path, batch_size, cache_dir, revision)
    else:
        raise ValueError(f"Unsupported dataset: {data_path}")
