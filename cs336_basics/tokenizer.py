import regex
from abc import ABC
from dataclasses import dataclass
from collections import defaultdict
import random
import torch
import tiktoken

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


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], dict[tuple[int, int], int]]:

    text = open(input_path, encoding="utf-8").read()
    print(f"Text befor pat: {text}")

    pattern = GPT2_TOKENIZER_REGEX  
    bpe_counts = defaultdict(int)
    idx = 0
    last_segment_bytes = None
    for match in regex.finditer(pattern, text):
        
        segment = match.group()
        print(segment)
        # 计算segment之间的bpe
        cur_segment_bytes = segment.encode("utf-8")
        if last_segment_bytes:
            bpe_counts[(last_segment_bytes, cur_segment_bytes)] += 1
        last_segment_bytes = cur_segment_bytes

        # 计算segment内部的bpe
        segment_list = list(map(int, segment.encode("utf-8")))
        for index1, index2 in zip(segment_list, segment_list[1:]):
            bpe_counts[(index1, index2)] += 1

        # Don't care across segment

        idx += 1

    for pair, v in bpe_counts.items():
        print(pair[0].decode("utf-8"), ' ', pair[1].decode("utf-8"), ' ', v) 

    indices = list(map(int, text.encode("utf-8")))
    merges: dict[tuple[int, int], int] = {} # index1, index2 => merged index



    vocab = {}
    vocab_idx = {}
    for st in special_tokens:
        vocab[len(vocab)] = st
    for i in range(256):
        vocab[len(vocab)] = bytes(i)

    vocab_idx = {v: k for k, v in vocab.items()}
    print(f"Original indices size is {len(indices)}\n")

    # Count the number of occurrences of each pair of tokens

    while len(vocab_idx) < vocab_size:
        counts = defaultdict(int)
        total_pair = 0
        for index1, index2 in zip(indices, indices[1:]):
            counts[(index1, index2)] += 1
            total_pair += 1

        if total_pair == len(counts):
            break

        # Find the most common pair
        counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

        for i, item in enumerate(counts):
            if item[1] <= 1:
                break
            
            pair = item[0]
            new_index = len(vocab_idx)
            if new_index >= vocab_size:
                break

            merges[pair] = new_index
            vocab_idx[new_index] = vocab_idx[pair[0]] + vocab_idx[pair[1]]

        indices = merge_indices(indices, merges)

    print(f"vocab size is {len(vocab_idx)}, final indices size is {len(indices)}")

    return (vocab_idx, merges)

    

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

    