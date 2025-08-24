import pathlib
import json
from bpe_tokenizer import train_bpe


DATA_PATH = (pathlib.Path(__file__).resolve().parents[2]) / "data"

def converts_bytes_to_str(obj):
    if isinstance(obj, dict):
        return {converts_bytes_to_str(key): converts_bytes_to_str(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [converts_bytes_to_str(item) for item in obj]
    elif isinstance(obj, set):
        return {converts_bytes_to_str(item) for item in obj}
    elif isinstance(obj, tuple):
        return tuple(converts_bytes_to_str(item) for item in obj)
    elif isinstance(obj, bytes):
        return obj.decode('utf-8', errors='replace').replace(" ", "Ġ")
    elif isinstance(obj, str):
        return obj.replace(" ", "Ġ")
    else:
        return obj

def train_tiny_stories_v2_valid(vocab_size, special_tokens, num_processes=4):
    input_path = DATA_PATH / "TinyStoriesV2-GPT4-valid.txt"
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        num_processes=num_processes
    )
    output_vocab_path = DATA_PATH / "TinyStoriesV2-GPT4-valid-vocab.json"
    output_merges_path = DATA_PATH / "TinyStoriesV2-GPT4-valid-merges.txt"
    with open(output_vocab_path, 'w', encoding="utf-8") as f:
        json.dump(converts_bytes_to_str(vocab), f, ensure_ascii=False, indent=4)
    with open(output_merges_path, 'w', encoding="utf-8") as f:
        for item in converts_bytes_to_str(merges):
            f.write(" ".join(item) + "\n")
        
def train_tiny_stories_v2_train(vocab_size, special_tokens, num_processes=4):
    input_path = DATA_PATH / "TinyStoriesV2-GPT4-train.txt"
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        num_processes=num_processes
    )
    output_vocab_path = DATA_PATH / "TinyStoriesV2-GPT4-train-vocab.json"
    output_merges_path = DATA_PATH / "TinyStoriesV2-GPT4-train-merges.txt"
    with open(output_vocab_path, 'w', encoding="utf-8") as f:
        json.dump(converts_bytes_to_str(vocab), f, ensure_ascii=False, indent=4)
    with open(output_merges_path, 'w', encoding="utf-8") as f:
        for item in converts_bytes_to_str(merges):
            f.write(" ".join(item) + "\n")

# find_chunk_boundaries time is 1.9683 seconds
# split chunks time is 2.4291 seconds
# chunks len is 1200
# bpe_count time is 78.9832 second
# merge time is 10536.7906 seconds
# bpe阶段内存: 15GB
# merge阶段内存: 13GB
def train_owt_valid(vocab_size, special_tokens, num_processes=4):
    input_path = DATA_PATH / "owt_valid.txt"
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        num_processes=num_processes
    )
    output_vocab_path = DATA_PATH / "owt_valid-vocab.json"
    output_merges_path = DATA_PATH / "owt_valid-merges.txt"
    with open(output_vocab_path, 'w', encoding="utf-8") as f:
        json.dump(converts_bytes_to_str(vocab), f, ensure_ascii=False, indent=4)
    with open(output_merges_path, 'w', encoding="utf-8") as f:
        for item in converts_bytes_to_str(merges):
            f.write(" ".join(item) + "\n")

# find_chunk_boundaries time is 51.3404 seconds
# split chunks time is 92.3496 seconds
# chunks len is 2400
# bpe_count time is 1046.1147 seconds
# bpe阶段内存：390 GB (2400chunk,128C)
# merge阶段内存: 100 GB
def train_owt_train(vocab_size, special_tokens, num_processes=4):
    input_path = DATA_PATH / "owt_train.txt"
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        num_processes=num_processes
    )
    output_vocab_path = DATA_PATH / "owt_train-vocab.json"
    output_merges_path = DATA_PATH / "owt_train-merges.txt"
    with open(output_vocab_path, 'w', encoding="utf-8") as f:
        json.dump(converts_bytes_to_str(vocab), f, ensure_ascii=False, indent=4)
    with open(output_merges_path, 'w', encoding="utf-8") as f:
        for item in converts_bytes_to_str(merges):
            f.write(" ".join(item) + "\n")

if __name__ == "__main__":

    special_tokens=["<|endoftext|>"]
    # train_tiny_stories_v2_valid(1000, special_tokens)
    # train_tiny_stories_v2_train(10000, special_tokens, 128)

    train_owt_valid(32000, special_tokens, 1200)
    # train_owt_train(32000, special_tokens, 2400)