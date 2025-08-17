import os
import multiprocessing as mp
from typing import BinaryIO, List
from collections import defaultdict
import regex


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_tokens: List[bytes],
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert all(isinstance(token, bytes) for token in split_special_tokens), "Must represent special tokens as bytestrings"

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

            # Find any of the special tokens in the mini chunk
            found_positions = []
            for token in split_special_tokens:
                found_at = mini_chunk.find(token)
                if found_at != -1:
                    found_positions.append(found_at)
            
            if found_positions:
                # Use the earliest found position
                chunk_boundaries[bi] = initial_position + min(found_positions)
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def process_chunk(args):
    """Process a chunk of text and return pre-tokenization counts"""
    chunk_text, pattern = args
    pre_token_cnt = defaultdict(int)
    
    for match in regex.finditer(pattern, chunk_text):
        segment = match.group()
        pre_token_cnt[segment] += 1
    
    return pre_token_cnt


def train_bpe(text: str, num_merges: int):  # @inspect string, @inspect num_merges
    # This is a placeholder - actual implementation would depend on the full BPE training algorithm
    pass
    
## Usage
def parallel_bpe_training(file_path: str, num_processes: int = 4):
    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, [b""])

    # Parallel implementation using multiprocessing
    chunks = []
    pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    with open(file_path, "rb") as f:
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append((chunk, pattern))
    
    # Process chunks in parallel
    with mp.Pool(num_processes) as pool:
        results = pool.map(process_chunk, chunks)
    
    # Combine results from all processes
    combined_counts = defaultdict(int)
    for pre_token_cnt in results:
        for token, count in pre_token_cnt.items():
            combined_counts[token] += count
    
    return combined_counts