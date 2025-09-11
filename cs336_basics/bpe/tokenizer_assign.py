import pathlib
import time
from bpe_tokenizer import BPETokenizer, get_compression_ratio
from collections import defaultdict
import numpy as np

DATA_PATH = (pathlib.Path(__file__).resolve().parents[2]) / "data"

OWT_TRAIN_PATH = DATA_PATH / "owt_train.txt"
OWT_VALID_PATH = DATA_PATH / "owt_valid.txt"
OWT_VOCAB_PATH = DATA_PATH / "owt_train-vocab.json"
OWT_MERGES_PATH = DATA_PATH / "owt_train-merges.txt"

TINY_TRAIN_PATH = DATA_PATH / "TinyStoriesV2-GPT4-train.txt"
TINY_VALID_PATH = DATA_PATH / "TinyStoriesV2-GPT4-valid.txt"
TINY_VOCAB_PATH = DATA_PATH / "TinyStoriesV2-GPT4-train-vocab.json"
TINY_MERGES_PATH = DATA_PATH / "TinyStoriesV2-GPT4-train-merges.txt"


def get_tokenizer():
    # owt_bpe_tokenizer = BPETokenizer.from_files(OWT_VOCAB_PATH, OWT_MERGES_PATH, ['<|endoftext|>'])
    tiny_bpe_tokenizer = BPETokenizer.from_files(TINY_VOCAB_PATH, TINY_MERGES_PATH, ['<|endoftext|>'])
    return None, tiny_bpe_tokenizer


def get_sample_text(text_path, top_n = 10):
    texts = []
    current_text = ""
    with open(text_path, 'r', encoding='utf-8') as f:
        for line in f:
            current_text += line
            while "<|endoftext|>" in current_text:
                split_texts = current_text.split("<|endoftext|>", 1)
                texts.append(split_texts[0] + "<|endoftext|>")
                current_text = split_texts[1]

            if len(texts) >= top_n:
                break

    return texts

def tokenizer_process(tokenizer, text):
    start_time = time.time()
    token_ids = tokenizer.encode(text)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return token_ids, elapsed_time, len(text.encode('utf-8'))

# Sample 1 OWT Compression Ratio: 4.3582, Tiny Compression Ratio: 3.3197
# Sample 2 OWT Compression Ratio: 4.8849, Tiny Compression Ratio: 3.2183
# Sample 3 OWT Compression Ratio: 4.4156, Tiny Compression Ratio: 3.0402
# Sample 4 OWT Compression Ratio: 4.1715, Tiny Compression Ratio:3.3868
# Sample 5 OWT Compression Ratio: 4.4766, Tiny Compression Ratio: 3.0141
# Sample 6 OWT Compression Ratio: 4.5674, Tiny Compression Ratio: 3.0219
# Sample 7 OWT Compression Ratio: 4.3920, Tiny Compression Ratio: 3.3578
# Sample 8 OWT Compression Ratio: 4.4537, Tiny Compression Ratio:2.7835
# Sample 9 OWT Compression Ratio: 4.2196, Tiny Compression Ratio: 2.8837
# Sample 10 OWT Compression Ratio: 4.8903, Tiny Compression Ratio: 3.1962
# Average OWT Tokenizer Throughout: 2551.51 bytes/second, Average Tiny Tokenizer Throughout: 8226.04 bytes/second
# Average OWT Compression Ratio: 4.4830, Average Tiny Compression Ratio: 3.1222
# Sample 1 Tiny Compression Ratio: 3.8615, OWT Compression Ratio: 3.7277
# Sample 2 Tiny Compression Ratio: 3.9302, OWT Compression Ratio: 3.5767
# Sample 3 Tiny Compression Ratio: 3.8824, OWT Compression Ratio: 3.9111
# Sample 4 Tiny Compression Ratio: 4.3168, OWT Compression Ratio: 4.2956
# Sample 5 Tiny Compression Ratio: 4.0886, OWT Compression Ratio: 4.1059
# Sample 6 Tiny Compression Ratio: 3.9153, OWT Compression Ratio: 3.9375
# Sample 7 Tiny Compression Ratio: 3.7588, OWT Compression Ratio:3.7588
# Sample 8 Tiny Compression Ratio: 3.8151, OWT Compression Ratio: 3.5748
# Sample 9 Tiny Compression Ratio: 3.9424, OWT Compression Ratio: 3.8728
# Sample 10 Tiny Compression Ratio: 4.1163, OWT Compression Ratio: 4.1355
# Average OWT Tokenizer Throughout: 1853.08 bytes/second, Average Tiny Tokenizer Throughout: 5993.01 bytes/second
# Average OWT Compression Ratio: 3.8896, Average Tiny (Compression Ratio: 3.9627
# Estimated OWT Tokenization Time: 478035384.43 seconds
# Estimated Tiny Tokenization Time: 147811607.35 seconds
def test_tokenizer():
    owt_bpe_tokenizer, tiny_bpe_tokenizer = get_tokenizer()
    owt_sample_texts = get_sample_text(OWT_TRAIN_PATH)
    tiny_sample_texts = get_sample_text(TINY_TRAIN_PATH)

    owt_ratios = []
    tiny_ratios = []
    # throughout = text bytes / time seconds
    owt_tokenizer_throughout = defaultdict(int)
    tiny_tokenizer_throughout = defaultdict(int)
    for i, text in enumerate(owt_sample_texts):
        owt_token_ids, elapsed_time, text_bytes_len = tokenizer_process(owt_bpe_tokenizer, text)
        owt_tokenizer_throughout['elapsed_time'] += elapsed_time
        owt_tokenizer_throughout['text_bytes'] += text_bytes_len

        tiny_token_ids, elapsed_time, text_bytes_len = tokenizer_process(tiny_bpe_tokenizer, text)
        tiny_tokenizer_throughout['elapsed_time'] += elapsed_time
        tiny_tokenizer_throughout['text_bytes'] += text_bytes_len

        owt_ratio = get_compression_ratio(text, owt_token_ids)
        tiny_ratio = get_compression_ratio(text, tiny_token_ids)
        owt_ratios.append(owt_ratio)
        tiny_ratios.append(tiny_ratio)
        print(f"Sample {i+1} OWT Compression Ratio: {owt_ratio:.4f}, Tiny Compression Ratio: {tiny_ratio:.4f}")

    avg_owt_throughput = owt_tokenizer_throughout['text_bytes'] / owt_tokenizer_throughout['elapsed_time']
    avg_tiny_throughput = tiny_tokenizer_throughout['text_bytes'] / tiny_tokenizer_throughout['elapsed_time']
    print(f"Average OWT Tokenizer Throughout: {avg_owt_throughput:.2f} bytes/second, Average Tiny Tokenizer Throughout: {avg_tiny_throughput:.2f} bytes/second")
    avg_owt_ratio = sum(owt_ratios) / len(owt_ratios)
    avg_tiny_ratio = sum(tiny_ratios) / len(tiny_ratios)
    print(f"Average OWT Compression Ratio: {avg_owt_ratio:.4f}, Average Tiny Compression Ratio: {avg_tiny_ratio:.4f}")
    owt_ratios = []
    tiny_ratios = []
    owt_tokenizer_throughout = defaultdict(int)
    tiny_tokenizer_throughout = defaultdict(int)

    for i, text in enumerate(tiny_sample_texts):
        owt_token_ids, elapsed_time, text_bytes = tokenizer_process(owt_bpe_tokenizer, text)
        owt_tokenizer_throughout['elapsed_time'] += elapsed_time
        owt_tokenizer_throughout['text_bytes'] += text_bytes

        tiny_token_ids, elapsed_time, text_bytes = tokenizer_process(tiny_bpe_tokenizer, text)
        tiny_tokenizer_throughout['elapsed_time'] += elapsed_time
        tiny_tokenizer_throughout['text_bytes'] += text_bytes

        owt_ratio = get_compression_ratio(text, owt_token_ids)
        tiny_ratio = get_compression_ratio(text, tiny_token_ids)
        owt_ratios.append(owt_ratio)
        tiny_ratios.append(tiny_ratio)
        print(f"Sample {i+1} Tiny Compression Ratio: {tiny_ratio:.4f}, OWT Compression Ratio: {owt_ratio:.4f}")

    avg_owt_throughput = owt_tokenizer_throughout['text_bytes'] / owt_tokenizer_throughout['elapsed_time']
    avg_tiny_throughput = tiny_tokenizer_throughout['text_bytes'] / tiny_tokenizer_throughout['elapsed_time']
    print(f"Average OWT Tokenizer Throughout: {avg_owt_throughput:.2f} bytes/second, Average Tiny Tokenizer Throughout: {avg_tiny_throughput:.2f} bytes/second")
    avg_owt_ratio = sum(owt_ratios) / len(owt_ratios)
    avg_tiny_ratio = sum(tiny_ratios) / len(tiny_ratios)
    print(f"Average OWT Compression Ratio: {avg_owt_ratio:.4f}, Average Tiny Compression Ratio: {avg_tiny_ratio:.4f}")

    # estimate 825GB text tokenization time
    pile_size = 825 * 1024**3  # 825GB in bytes
    owt_tokenization_time = pile_size / avg_owt_throughput
    tiny_tokenization_time = pile_size / avg_tiny_throughput
    print(f"Estimated OWT Tokenization Time: {owt_tokenization_time:.2f} seconds")
    print(f"Estimated Tiny Tokenization Time: {tiny_tokenization_time:.2f} seconds")


def tokenizer_unit_test():

    tiny_bpe_tokenizer = BPETokenizer.from_files(TINY_VOCAB_PATH, TINY_MERGES_PATH, ['<|endoftext|>'])
    print(f"tokenizer vocab top 257 items: {list(tiny_bpe_tokenizer.vocab_idx2bytes.items())[:257]}")

    text = """
Once upon a time there was a little girl. She lived in a pretty house by the sea.
The little girl was sad. She was so sad that she couldn't even play with her friends. Every day she would sit by herself and cry.
One day her mom found out. She asked her daughter why she was sad. The little girl's eyes filled with tears and she said “I don't know the answers to my quiz”.
Her mom hugged her and said “It's ok, you don't have to know the answers to your quiz. We just have to find a way to help you heal."
The little girl was so relieved and hugged her mom tightly.
From then on, the little girl was no longer ashamed. She practiced every day and soon enough, she knew the answers to her quiz!
<|endoftext|>"""

    tiny_token_ids = tiny_bpe_tokenizer.encode(text)
    print(f"Tiny Token IDs: {tiny_token_ids}")
    print(f"np array: {np.array(tiny_token_ids, dtype=np.uint16)}")

def tokenizer_file(tokenizer, file_path, output_path):
    print(f"Tokenizing {file_path} to {output_path}...")
    all_token_ids = np.array([], dtype=np.uint16)
    start_time = time.time()
    
    current_text = ""
    processed_bytes = 0
    with open(file_path, 'r', encoding='utf-8') as f_in, open(output_path, 'w', encoding='utf-8') as f_out:
        is_process = False
        current_bytes = 0
        for line in f_in:
            current_text += line
            current_bytes += len(line.encode("utf-8"))

            if current_bytes >= 10 * 1024 * 1024:  # Process every 10MB
                is_process = True

            if is_process and "<|endoftext|>" in line:
                split_line = line.split("<|endoftext|>", 1)
                text_to_tokenize = current_text + split_line[0] + "<|endoftext|>"
                print(text_to_tokenize)
                current_text = split_line[1]
                token_ids = tokenizer.encode(text_to_tokenize)
                try:
                    all_token_ids = np.concatenate((all_token_ids, np.array(token_ids, dtype=np.uint16)))
                except Exception as e:
                    print(f"Error processing text: {text_to_tokenize}, token_ids so far: {token_ids}")
                    print(e)
                processed_bytes += current_bytes
                print(f"Processed {processed_bytes / (1024**2):.2f} MB", end='\r')

                current_bytes = len(current_text.encode("utf-8"))
                is_process = False


        end_time = time.time()
        elapsed_time = end_time - start_time
        f_out.write(','.join(map(str, all_token_ids)))

    print(f"Tokenization {file_path} to {output_path} completed in {elapsed_time:.2f} seconds.")
        

def get_numpy_array_from_file(tokenized_file_path):
    with open(tokenized_file_path, 'r', encoding='utf-8') as f:
        token_ids_str = f.read().strip()
        if token_ids_str:
            token_ids = np.array(list(map(int, token_ids_str.split(','))), dtype=np.uint16)
        else:
            token_ids = np.array([], dtype=np.uint16)
    return token_ids

def main():
    owt_bpe_tokenizer, tiny_bpe_tokenizer = get_tokenizer()
    tokenizer_file(tiny_bpe_tokenizer, TINY_VALID_PATH, DATA_PATH / "tiny_valid_tokenized.txt")
    tokenizer_file(tiny_bpe_tokenizer, TINY_TRAIN_PATH, DATA_PATH / "tiny_train_tokenized.txt")
    # tokenizer_file(owt_bpe_tokenizer, OWT_VALID_PATH, DATA_PATH / "owt_valid_tokenized.txt")

    test_np = get_numpy_array_from_file(DATA_PATH / "tiny_valid_tokenized.txt")
    print(f"Loaded {len(test_np)} tokens from tokenized file.")

if __name__ == "__main__":
    # tokenizer_unit_test()
    main()
    # test_tokenizer()