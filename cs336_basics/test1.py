

import torch

def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    print(bytestring)
    for b in bytestring:
        print(b, ' ', bytes([b]), ' ', bytes([b]).decode("utf-8"))
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

if __name__ == "__main__":

    # print(decode_utf8_bytes_to_str_wrong("hello! こんにちは!".encode("utf-8")))


    for i in range(256):
        for j in range(256):
            b = bytes([i, j])
            try:
                b.decode("utf-8")
            except Exception:
                print(f"Cannot decode: {b}")