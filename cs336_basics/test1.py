

import torch
import regex as re

def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    print(bytestring)
    for b in bytestring:
        print(b, ' ', bytes([b]), ' ', bytes([b]).decode("utf-8"))
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

if __name__ == "__main__":
    string = "s12121212"
    special_tokens = []
    print(re.split("|".join(map(re.escape, special_tokens)), string))

        
