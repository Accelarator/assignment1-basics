import regex as re
from abc import ABC
from dataclasses import dataclass
from collections import defaultdict
import random
import torch
import tiktoken
import os
from typing import BinaryIO
import multiprocessing as mp
import traceback
from tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode
import time

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
    
def initial_vocab(special_tokens):
    vocab_idx2bytes = {}
    vocab_bytes2idx = {}
    for st in special_tokens:
        vocab_idx2bytes[len(vocab_idx2bytes)] = st.encode("utf-8")
    for i in range(256):
        vocab_idx2bytes[len(vocab_idx2bytes)] = bytes([i])

    vocab_bytes2idx = {v: k for k, v in vocab_idx2bytes.items()}
    return vocab_idx2bytes, vocab_bytes2idx

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_tokens: list[bytes],
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert all(isinstance(token, bytes) for token in split_special_tokens), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size
    
    print(f"initial chunk_boundaries is {chunk_boundaries}")

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
            found_at = -1
            for special_token in split_special_tokens:
                fa = mini_chunk.find(special_token)
                if fa != -1 and (fa < found_at or found_at == -1):
                    found_at = fa
                    # print(f"fa is {fa}, found_at is {found_at}")

            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    print(f"final chunk_boundaries is {chunk_boundaries}")

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))
def merge_pair(pre_token_cnt, pre_token_indices, bpe_counts, new_bytes, pair, new_index):

    for segment, indices in pre_token_indices.items():
        
        if new_bytes in segment:
            cnt = pre_token_cnt[segment]
            has_sub = []
            new_indices = []
            i = 0
            while i < len(indices):
                if i + 1 < len(indices) and indices[i] == pair[0] and indices[i+1] == pair[1]:
                    new_indices.append(new_index)
                    if i - 1 >= 0 and (i - 1, i) not in has_sub:
                        bpe_counts[(indices[i-1], pair[0])] -= cnt
                        has_sub.append((i - 1, i))
                    if i + 2 < len(indices) and (i + 1, i + 2) not in has_sub:
                        bpe_counts[(pair[1], indices[i+2])] -= cnt
                        has_sub.append((i + 1, i + 2))
                    i += 2
                else:
                    new_indices.append(indices[i])
                    i += 1

            # 更新bpe_counts
            bpe_counts[pair] = 0  
            has_add = [] 
            for i in range(len(new_indices)):
                if new_indices[i] == new_index:
                    if i - 1 >= 0 and (i - 1, i) not in has_add:
                        bpe_counts[(new_indices[i-1], new_index)] += cnt
                        has_add.append((i - 1, i))
                    if i + 1 < len(new_indices) and (i, i + 1) not in has_add:
                        bpe_counts[(new_index, new_indices[i+1])] += cnt
                        has_add.append((i, i + 1))
            pre_token_indices[segment] = new_indices

def get_bpe_counts(pre_token_cnt, pre_token_indices):

    bpe_counts = defaultdict(int)
    for segment, indices in pre_token_indices.items():
        cnt = pre_token_cnt[segment]
        for index1, index2 in zip(indices, indices[1:]):
            bpe_counts[(index1, index2)] += cnt

            # if index1 > 128 or index2 > 128:
            #     print(f"大于128的segment是: {segment}: {indices}")

    return bpe_counts

def get_pre_tokens(chunks: list[str], 
                   vocab_bytes2idx: dict[bytes, int], 
                   special_tokens: list[str]| None = None
                   ):
    pattern = GPT2_TOKENIZER_REGEX 
    pre_token_cnt = defaultdict(int) # token str => token cnt
    pre_token_indices = {} # token str => token indices
    pre_token_seg = [] # list[tuple(str, token_list)]
    for text in chunks:
        token_list = []
        if special_tokens and text in special_tokens:
            # print(f"{vocab_bytes2idx.get(text.encode("utf-8"))}")
            pre_token_seg.append((text, [vocab_bytes2idx.get(text.encode("utf-8"))]))
        else:
            for match in re.finditer(pattern, text):
                segment = match.group().encode("utf-8")
                pre_token_cnt[segment] += 1
                pre_token_indices[segment] = [vocab_bytes2idx.get(bytes([b])) for b in segment]
                token_list.append(segment)
            pre_token_seg.append((text, token_list))
    
    return pre_token_cnt, pre_token_indices, pre_token_seg

def process_pre_tokens(args):
    text, special_tokens, vocab_bytes2idx = args

    start_time = time.time()
    
    chunks = re.split("|".join(map(re.escape, special_tokens)), text)
    # print(chunks)

    pre_token_cnt, pre_token_indices, _ = get_pre_tokens(chunks, vocab_bytes2idx, special_tokens)
    bpe_counts = get_bpe_counts(pre_token_cnt, pre_token_indices)
    end_time = time.time()
    # print(f"bpe_count time is {end_time - start_time:.4f} seconds")

    return bpe_counts, pre_token_cnt, pre_token_indices

def train_bpe(input_path, vocab_size, special_tokens, num_processes = 4):
    split_special_tokens = [special_token.encode("utf-8") for special_token in special_tokens]

    start_time = time.time()
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, split_special_tokens)
    end_time = time.time()
    print(f"find_chunk_boundaries time is {end_time - start_time:.4f} seconds")

    vocab_idx2bytes, vocab_bytes2idx = initial_vocab(special_tokens)

    start_time = time.time()
    chunks = []
    with open(input_path, "rb") as f:
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append((chunk, special_tokens, vocab_bytes2idx))
    end_time = time.time()
    print(f"split chunks time is {end_time - start_time:.4f} seconds")

    start_time = time.time()

    print(f"chunks len is {len(chunks)}")
    with mp.Pool(num_processes) as pool:
        results = pool.map(process_pre_tokens, chunks)
    
    bpe_counts = defaultdict(int) # index1, index2 => cnt
    pre_token_cnt = defaultdict(int) # token => token cnt
    pre_token_indices = {} # token => token indices
    for res in results:
        for bpe_pair, cnt in res[0].items():
            bpe_counts[bpe_pair] += cnt
        
        for pre_token, cnt in res[1].items():
            pre_token_cnt[pre_token] += cnt

        pre_token_indices.update(res[2])

    # print(f"vocab_idx2bytes: {vocab_idx2bytes}, vocab_bytes2idx: {vocab_bytes2idx}")
    merges: dict[tuple[int, int], int] = {} # index1, index2 => merged index
    return_merges = []

    end_time = time.time()
    print(f"bpe_count time is {end_time - start_time:.4f} seconds")
    start_time = time.time()

    
    log_pair = [(b' g', b'ive'), (b'\n', b'\n')]

    while len(vocab_idx2bytes) < vocab_size:
        if len(vocab_idx2bytes) >= vocab_size:
            break

        # bpe_counts = get_bpe_counts(pre_token_cnt, pre_token_indices)
        # 优先按数量倒序排序，否则按pair[0]字典序排序
        max_item = max(bpe_counts.items(), key=lambda x: (x[1], vocab_idx2bytes[x[0][0]], vocab_idx2bytes[x[0][1]]))
        if max_item[1] <= 0:
            break

        # new_bpe_counts = sorted(bpe_counts.items(), key=lambda x: (x[1], vocab_idx2bytes[x[0][0]], vocab_idx2bytes[x[0][1]]), reverse=True)
        # print(f"BPE counts sorted: {[((vocab_idx2bytes[pair[0]], vocab_idx2bytes[pair[1]]), v) for pair, v in new_bpe_counts][:10]}")
        # print(f"Sorted item is {new_bpe_counts[0]}, manua sort cur pair is {cur_pair}")
        new_index = len(vocab_idx2bytes)
        
        max_pair = max_item[0]
        max_cnt = max_item[1]
        index1 = max_pair[0]
        index2 = max_pair[1]
        new_bytes = vocab_idx2bytes[index1] + vocab_idx2bytes[index2]
        # print(f"vocab_idx2bytes[index1] is {vocab_idx2bytes[index1]}, vocab_idx2bytes[index2] is {vocab_idx2bytes[index2]}, new_bytes is {new_bytes}")
        # print(f"new_bytes is {new_bytes}, pair is {max_pair}, cnt is {max_item[1]}")

        merges[(index1, index2)] = new_index
        return_merges.append((vocab_idx2bytes[index1], vocab_idx2bytes[index2]))
        vocab_bytes2idx[new_bytes] = new_index
        vocab_idx2bytes[new_index] = new_bytes

        record_before_cnt = []
        for pair in log_pair:
            try:
                # print(f"{pair[0]}: {vocab_bytes2idx[pair[0]]}, {pair[1]}: {vocab_bytes2idx[pair[1]]}, {bpe_counts[vocab_bytes2idx[pair[0]], vocab_bytes2idx[pair[1]]]}")
                record_before_cnt.append(bpe_counts[vocab_bytes2idx[pair[0]], vocab_bytes2idx[pair[1]]])
            except Exception as e:
                record_before_cnt.append(0)

        # 更新bpe_counts
        merge_pair(pre_token_cnt, pre_token_indices, bpe_counts, new_bytes, max_pair, new_index)
       
    end_time = time.time()
    print(f"merge time is {end_time - start_time:.4f} seconds")

    return (vocab_idx2bytes, return_merges)
    


def get_compression_ratio(string: str, indices: list[int]) -> float:
    """Given `string` that has been tokenized into `indices`."""
    num_bytes = len(bytes(string, encoding="utf-8"))
    num_tokens = len(indices)
    return num_bytes / num_tokens


class BPETokenizer(Tokenizer):
    """BPE tokenizer given a set of merges and a vocabulary."""
    def __init__(self, 
                vocab: dict[int, bytes],
                merges: list[tuple[bytes, bytes]],
                special_tokens: list[str] | None = None):
        
        self.vocab_idx2bytes = vocab
        self.vocab_bytes2idx = {v: k for k, v in vocab.items()}
        self.merges = merges
        if special_tokens:
            self.special_tokens = sorted(special_tokens, key=lambda x: len(x), reverse=True)
        else:
            self.special_tokens = special_tokens

    def encode(self, string: str) -> list[int]:

        if self.special_tokens:
            escaped_tokens = [re.escape(token) for token in self.special_tokens]  # 转义特殊字符
            special_pattern = "(" + "|".join(escaped_tokens) + ")"
            chunks = re.split(special_pattern, string)
        else:
            chunks = [string]
        # print(f"chunks is {chunks}")
        _, pre_token_indices, pre_token_seg = get_pre_tokens(chunks, self.vocab_bytes2idx, self.special_tokens)

        # print(f"pre_token_seg is {pre_token_seg}")
        # print(f"pre_token_indices is {pre_token_indices}")
        for merge in self.merges:
            new_bytes = merge[0] + merge[1]

            for segment, indices in pre_token_indices.items():
                if segment == b'':
                    continue

                if new_bytes in segment:

                    new_index = self.vocab_bytes2idx[new_bytes]
                    index1 = self.vocab_bytes2idx[merge[0]]
                    index2 = self.vocab_bytes2idx[merge[1]]

                    new_indices = []
                    i = 0
                    while i < len(indices):
                        if i + 1 < len(indices) and indices[i] == index1 and indices[i+1] == index2:
                            new_indices.append(new_index)
                            i += 2
                        else:
                            new_indices.append(indices[i])
                            i += 1
                    pre_token_indices[segment] = new_indices

        # print(f"result pre_token_indices {pre_token_indices}")
        res_indices = []
        for pre_token_tuple in pre_token_seg:
            chunk = pre_token_tuple[0]
            pre_tokens = pre_token_tuple[1]
            if chunk == "":
                continue
            if self.special_tokens and chunk in self.special_tokens:
                res_indices.extend(pre_tokens)
            else:
                for pre_token in pre_tokens:
                    res_indices.extend(pre_token_indices[pre_token])

        return res_indices
    
    def decode(self, indices: list[int]) -> str:
        # print(f"indices is {indices}")
        bytes_list = []
        for item in indices:
            if isinstance(item, list):
                bytes_list.extend(list(map(self.vocab_idx2bytes.get, item)))
            else:
                bytes_list.append(self.vocab_idx2bytes[item])
                
        string = b"".join(bytes_list).decode("utf-8", errors='replace')
        return string

    def encode_iterable(self, iterable) -> list[list[int]]:
        res_encode_ids = []
        for chunk in iterable:
            encode_ids = self.encode(chunk)
            res_encode_ids.append(encode_ids)
        return res_encode_ids
