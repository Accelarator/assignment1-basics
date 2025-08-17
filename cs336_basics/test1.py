

import torch

def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    print(bytestring)
    for b in bytestring:
        print(b, ' ', bytes([b]), ' ', bytes([b]).decode("utf-8"))
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

if __name__ == "__main__":

    str = "sabcasldf"
    print("".encode("utf-8"))

    print(str[0:2])
    print(str[3:5])
    print(str[2:3])
    print(str[5:6])


    vocab_bytes2idx = {}
    vocab_bytes2idx[b'b'] = 11
    vocab_bytes2idx[b'bc'] = 111 
    str_b = "b"
    assert str_b.encode("utf-8") in vocab_bytes2idx
    str_bc = "bc"
    assert str_bc.encode("utf-8") in vocab_bytes2idx

        
