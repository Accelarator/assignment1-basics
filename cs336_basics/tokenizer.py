import regex
from abc import ABC
from dataclasses import dataclass
from collections import defaultdict
import random
import torch
import tiktoken
import os
from typing import BinaryIO

GPT2_TOKENIZER_REGEX = \
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer(ABC):
    """Abstract interface for a tokenizer."""
    def encode(self, string: str) -> list[int]:
        raise NotImplementedError
    def decode(self, indices: list[int]) -> str:
        raise NotImplementedError

@dataclass(frozen=True)
class BPETokenizerParams:
    """All you need to specify a BPETokenizer"""
    vocab: dict[int, bytes]  # index -> bytes
    merges: dict[tuple[int, int], int] # index1,index2 -> new_index

class CharacterTokenizer(Tokenizer):
    """Represent a string as a sequence of Unicode code points."""
    def encode(self, string: str) -> list[int]:
        return list(map(ord, string))
    
    def decode(self, indices: list[int]) -> str:
        return "".join(map(chr, indices))
    

class ByteTokenizer(Tokenizer):
    """Represent a string as a sequence of bytes."""
    def encode(self, string: str) -> list[int]:
        string_bytes = string.encode("utf-8")
        indices = list(map(int, string_bytes))
        return indices
    
    def decode(self, indices: list[int]) -> str:
        string_bytes = bytes(indices)
        string = string_bytes.decode("utf-8")
        return string
    
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def merge(indices: list[int], pair: tuple[int, int], new_index: int) -> list[int]:
    """Return `indices`, but with all instances of `pair` replaced with `new_index`."""
    new_indices = []
    i = 0
    while i < len(indices):
        if i + 1 < len(indices) and indices[i] == pair[0] and indices[i+1] == pair[1]:
            new_indices.append(new_index)
            i += 2
        else:
            new_indices.append(indices[i])
            i += 1
    return new_indices

class BPETokenizer(Tokenizer):
    """BPE tokenizer given a set of merges and a vocabulary."""
    def __init__(self, params: BPETokenizerParams):
        self.params = params

    def encode(self, string: str) -> list[int]:
        indices = list(map(int, string.encode("utf-8")))
        # Note: this is a very slow implementation
        for pair, new_index in self.params.merges.items():
            indices = merge(indices, pair, new_index)
        return indices
    
    def decode(self, indices: list[int]) -> str:
        bytes_list = list(map(self.params.vocab.get, indices))
        string = b"".join(bytes_list).decode("utf-8")
        return string
    

def train_bpe_old(string: str, num_merges: int) -> BPETokenizerParams:  # @inspect string, @inspect num_merges
    # Start with the list of bytes of string.
    indices = list(map(int, string.encode("utf-8")))  # @inspect indices
    merges: dict[tuple[int, int], int] = {}  # index1, index2 => merged index
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}  # index -> bytes
    for i in range(num_merges):
        # Count the number of occurrences of each pair of tokens
        counts = defaultdict(int)
        for index1, index2 in zip(indices, indices[1:]):  # For each adjacent pair
            counts[(index1, index2)] += 1  # @inspect counts
        # Find the most common pair.
        pair = max(counts, key=counts.get)  # @inspect pair
        index1, index2 = pair
        # Merge that pair.
        new_index = 256 + i  # @inspect new_index
        merges[pair] = new_index  # @inspect merges
        vocab[new_index] = vocab[index1] + vocab[index2]  # @inspect vocab
        indices = merge(indices, pair, new_index)  # @inspect indices
    return BPETokenizerParams(vocab=vocab, merges=merges)


def merge_indices(indices: list[int], merges: dict[tuple[int, int], int]) -> list[int]:
    if_merge = True
    while if_merge:
        if_merge = False
        new_indices = []
        i = 0
        # print(f"Merge indices, current size is {len(indices)}")
        while i < len(indices):
            if i + 1 < len(indices) and (indices[i], indices[i+1]) in merges:
                new_indices.append(merges[indices[i], indices[i+1]])
                i += 2
                if_merge = True
            else:
                new_indices.append(indices[i])
                i += 1
        indices = new_indices
    return indices



def merge_pair(pre_token_cnt, pre_token_indices, bpe_counts, new_str, pair, new_index):

    for segment, indices in pre_token_indices.items():
        
        if new_str in segment:
            cnt = pre_token_cnt[segment]
            new_indices = []
            i = 0
            while i < len(indices):
                if i + 1 < len(indices) and indices[i] == pair[0] and indices[i+1] == pair[1]:
                    new_indices.append(new_index)
                    i += 2
                else:
                    new_indices.append(indices[i])
                    i += 1

            # 更新bpe_counts
            bpe_counts[pair] = -1
            for i in range(len(new_indices)):
                if new_indices[i] == new_index:
                    if i - 1 >= 0:
                        bpe_counts[(new_indices[i-1], pair[0])] -= cnt
                        bpe_counts[(new_indices[i-1], new_index)] += cnt
                    if i + 1 < len(new_indices):
                        bpe_counts[(pair[1], new_indices[i+1])] -= cnt
                        bpe_counts[(new_index, new_indices[i+1])] += cnt
            
            pre_token_indices[segment] = new_indices

def get_bpe_counts(pre_token_cnt, pre_token_indices):

    bpe_counts = defaultdict(int)
    for segment, indices in pre_token_indices.items():
        cnt = pre_token_cnt[segment]
        for index1, index2 in zip(indices, indices[1:]):
            bpe_counts[(index1, index2)] += cnt

    return bpe_counts

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], dict[tuple[int, int], int]]:

    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")

    text = open(input_path, encoding="utf-8").read()
    # print(f"Text befor pat: {text}")

    vocab_idx2bytes = {}
    vocab_bytes2idx = {}
    for st in special_tokens:
        vocab_idx2bytes[len(vocab_idx2bytes)] = st.encode("utf-8")
    for i in range(256):
        vocab_idx2bytes[len(vocab_idx2bytes)] = bytes([i])

    vocab_bytes2idx = {v: k for k, v in vocab_idx2bytes.items()}

    pattern = GPT2_TOKENIZER_REGEX  
    pre_token_cnt = defaultdict(int) # token str => token cnt
    pre_token_indices = {} # token str => token indices
    for match in regex.finditer(pattern, text):
        segment = match.group()
        pre_token_cnt[segment] += 1
        pre_token_indices[segment] = list(map(lambda x: x + len(special_tokens), segment.encode("utf-8")))

    bpe_counts = get_bpe_counts(pre_token_cnt, pre_token_indices)
    merges: dict[tuple[int, int], int] = {} # index1, index2 => merged index
    return_merges = []
    
    while len(vocab_idx2bytes) < vocab_size:
        if len(vocab_idx2bytes) >= vocab_size:
            break

        # 优先按数量倒序排序，否则按pair[0]字典序排序
        max_cnt = 0
        lst_key_bytes = "".encode("utf-8")
        lst_value_bytes = "".encode("utf-8")
        cur_pair = None
        for pair, cnt in bpe_counts.items():
            cur_key_bytes = vocab_idx2bytes[pair[0]]
            cur_value_bytes = vocab_idx2bytes[pair[1]]
            is_replace = False
            if cnt > max_cnt:
                is_replace = True
            elif cnt == max_cnt:
                if cur_key_bytes > lst_key_bytes:
                    is_replace = True
                elif cur_key_bytes == lst_key_bytes:
                    if cur_value_bytes > lst_value_bytes:
                        is_replace = True

            if is_replace:
                cur_pair = pair
                max_cnt = cnt
                lst_key_bytes = cur_key_bytes
                lst_value_bytes = cur_value_bytes

        # new_bpe_counts = sorted(bpe_counts.items(), key=lambda x: (x[1], vocab_idx2bytes[x[0][0]], vocab_idx2bytes[x[0][1]]), reverse=True)
        # print(f"BPE counts sorted: {[((vocab_idx2bytes[pair[0]], vocab_idx2bytes[pair[1]]), v) for pair, v in new_bpe_counts][:10]}")
        # print(f"Sorted item is {new_bpe_counts[0]}, manua sort cur pair is {cur_pair}")
        new_index = len(vocab_idx2bytes)
        
        max_pair = cur_pair
        index1 = max_pair[0]
        index2 = max_pair[1]
        new_bytes = vocab_idx2bytes[index1] + vocab_idx2bytes[index2]
        # print(f"vocab_idx2bytes[index1] is {vocab_idx2bytes[index1]}, vocab_idx2bytes[index2] is {vocab_idx2bytes[index2]}, new_bytes is {new_bytes}")

        merges[(index1, index2)] = new_index
        return_merges.append((vocab_idx2bytes[index1], vocab_idx2bytes[index2]))
        vocab_bytes2idx[new_bytes] = new_index
        vocab_idx2bytes[new_index] = new_bytes
        new_str = new_bytes.decode("utf-8")

        # 更新bpe_counts
        merge_pair(pre_token_cnt, pre_token_indices, bpe_counts, new_str, max_pair, new_index)
                  
    return (vocab_idx2bytes, return_merges)

    

def get_compression_ratio(string: str, indices: list[int]) -> float:
    """Given `string` that has been tokenized into `indices`."""
    num_bytes = len(bytes(string, encoding="utf-8"))
    num_tokens = len(indices)
    return num_bytes / num_tokens

def get_gpt2_tokenizer():
    # Code: https://github.com/openai/tiktoken
    # You can use cl100k_base for the gpt3.5-turbo or gpt4 tokenizer
    return tiktoken.get_encoding("gpt2")


def tokenization_examples():
    # To get a feel for how tokenizers work, play with this  interactive site
    # Observations
    # 
    # A word and its preceding space are part of the same token (e.g., " world").
    #     
    # A word at the beginning and in the middle are represented differently (e.g., "hello hello").
    #     
    # Numbers are tokenized into every few digits.
    # Here's the GPT-2 tokenizer from OpenAI (tiktoken) in action.
    tokenizer = get_gpt2_tokenizer()
    string = "Hello, 🌍! 你好!"  # @inspect string
    # Check that encode() and decode() roundtrip:
    indices = tokenizer.encode(string)  # @inspect indices
    reconstructed_string = tokenizer.decode(indices)  # @inspect reconstructed_string
    assert string == reconstructed_string
    compression_ratio = get_compression_ratio(string, indices)  # @inspect compression_ratio

    print(f"indices: {indices}")
    print(f"reconstructed_string: {reconstructed_string}")
    print(f"compression_ratio: {compression_ratio}")

def character_tokenizer():
    # Character-based tokenization
    # A Unicode string is a sequence of Unicode characters.
    # Each character can be converted into a code point (integer) via ord.
    assert ord("a") == 97
    assert ord("🌍") == 127757
    # It can be converted back via chr.
    assert chr(97) == "a"
    assert chr(127757) == "🌍"
    # Now let's build a Tokenizer and make sure it round-trips:
    tokenizer = CharacterTokenizer()
    string = "Hello, 🌍! 你好!"  # @inspect string
    indices = tokenizer.encode(string)  # @inspect indices
    reconstructed_string = tokenizer.decode(indices)  # @inspect reconstructed_string
    assert string == reconstructed_string
    # There are approximately 150K Unicode characters.  [Wikipedia]
    vocabulary_size = max(indices) + 1  # This is a lower bound @inspect vocabulary_size
    # Problem 1: this is a very large vocabulary.
    # Problem 2: many characters are quite rare (e.g., 🌍), which is inefficient use of the vocabulary.
    compression_ratio = get_compression_ratio(string, indices)  # @inspect compression_ratio

    print(f"indices: {indices}")
    print(f"reconstructed_string: {reconstructed_string}")
    print(f"compression_ratio: {compression_ratio}")

def byte_tokenizer():
    # Byte-based tokenization
    # Unicode strings can be represented as a sequence of bytes, which can be represented by integers between 0 and 255.
    # The most common Unicode encoding is  UTF-8
    # Some Unicode characters are represented by one byte:
    assert bytes("a", encoding="utf-8") == b"a"
    # Others take multiple bytes:
    assert bytes("🌍", encoding="utf-8") == b"\xf0\x9f\x8c\x8d"
    # Now let's build a Tokenizer and make sure it round-trips:
    tokenizer = ByteTokenizer()
    string = "Hello, 🌍! 你好!"  # @inspect string
    indices = tokenizer.encode(string)  # @inspect indices
    reconstructed_string = tokenizer.decode(indices)  # @inspect reconstructed_string
    assert string == reconstructed_string
    # The vocabulary is nice and small: a byte can represent 256 values.
    vocabulary_size = 256  # @inspect vocabulary_size
    # What about the compression rate?
    compression_ratio = get_compression_ratio(string, indices)  # @inspect compression_ratio
    assert compression_ratio == 1
    # The compression ratio is terrible, which means the sequences will be too long.
    # Given that the context length of a Transformer is limited (since attention is quadratic), this is not looking great...

    print(f"indices: {indices}")
    print(f"reconstructed_string: {reconstructed_string}")
    print(f"compression_ratio: {compression_ratio}")


def word_tokenizer():
    # Word-based tokenization
    # Another approach (closer to what was done classically in NLP) is to split strings into words.
    string = "I'll say supercalifragilisticexpialidocious!"
    segments = regex.findall(r"\w+|.", string)  # @inspect segments
    # This regular expression keeps all alphanumeric characters together (words).
    # Here is a fancier version:
    pattern = GPT2_TOKENIZER_REGEX  # @inspect pattern
    segments = regex.findall(pattern, string)  # @inspect segments
    # To turn this into a Tokenizer, we need to map these segments into integers.
    # Then, we can build a mapping from each segment into an integer.
    # But there are problems:
    # 
    # The number of words is huge (like for Unicode characters).
    #     
    # Many words are rare and the model won't learn much about them.
    #     
    # This doesn't obviously provide a fixed vocabulary size.
    # New words we haven't seen during training get a special UNK token, which is ugly and can mess up perplexity calculations.
    vocabulary_size = "Number of distinct segments in the training data"
    compression_ratio = get_compression_ratio(string, segments)  # @inspect compression_ratio

    print(f"segments: {segments}")
    print(f"compression_ratio: {compression_ratio}")


def bpe_tokenizer():

    # Training the tokenizer
    string = "the cat in the hat"
    params = train_bpe(string, num_merges=3)

    # Using the tokenizer
    tokenizer = BPETokenizer(params)
    string = "the quick brown fox"
    indices = tokenizer.encode(string)
    reconstructed_string = tokenizer.decode(indices)

    print(f"indices: {indices}")
    print(f"reconstructed_string: {reconstructed_string}")

if __name__ == "__main__":
    # tokenization_examples()
    # character_tokenizer()
    # byte_tokenizer()
    # word_tokenizer()
    bpe_tokenizer()

    