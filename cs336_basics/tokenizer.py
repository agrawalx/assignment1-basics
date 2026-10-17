import regex as re
import copy
from typing import BinaryIO

def BPE_tokenizer(
    file_path: str,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]: 
    "vocab, merges"
    
    for i in special_tokens:
        assert isinstance(i, str), "Must represent special token as a str"
    sorted_special_tokens = sorted(special_tokens, key=len, reverse = True)
    next_token_id = 256
    pattern = "|".join(re.escape(token) for token in sorted_special_tokens)
    with open(file_path, "rb") as file:
        text = file.read().decode("utf-8")
    if len(special_tokens) != 0:
        parts = re.split(pattern, text)
    else:
        parts = [text]
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    tokens: dict[tuple[bytes, ...], int] = {}
    token_list = []
    for part in parts:
        words = re.finditer(PAT, part)
        for word in words:
            word = word.group().encode("utf-8")
            word_len = len(word)
            key: tuple[bytes, ...] = ()
            for i in range(0, word_len):
                key = key + (word[i:i+1],)
            if key in tokens:
                tokens[key] += 1 
            else:
                tokens[key] =1 
    # pre-tokenization complete here, the tokens dict contains all the tokens with their counts. 
    # next step is to create pairs and start merging. 
    for token, count in tokens.items():
        token_list.append((token, count))
    num_loops = vocab_size - 256 - len(special_tokens)
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])
    for i in sorted_special_tokens:
        vocab[next_token_id] = i.encode("utf-8")
        next_token_id+=1
    merges = []
    pairs_dict: dict[tuple[bytes, bytes], set[int]] = {} # (bytes, bytes) -> [int]
    pairs_count: dict[tuple[bytes, bytes], int] = {}
    for i in range(len(token_list)):
        (token, count)= token_list[i]
        for j in range(len(token)-1):
            pair = (token[j], token[j+1])
            existing_set = pairs_dict.get(pair, None)
            if existing_set is not None:
                existing_set.add(i)
                pairs_count[pair] += count
            else:
                pairs_dict[pair] = {i}
                pairs_count[pair] = count

    for i in range(num_loops):
        if len(pairs_count) == 0:
            break
        max_pair = max(pairs_count, key=lambda k: (pairs_count[k], k))
        vocab[next_token_id] = max_pair[0] + max_pair[1]
        next_token_id+=1
        merges.append(max_pair)
        tokens_index = copy.deepcopy(pairs_dict[max_pair])
        for j in tokens_index:
            token = token_list[j][0]
            count = token_list[j][1]
            new_token = []
            k = 0
            while k<len(token):
                if k!= len(token)-1 and (token[k] == max_pair[0] and token[k+1] == max_pair[1]):
                    new_token.append(token[k] + token[k+1])
                    k += 2
                else:
                    new_token.append(token[k])
                    k += 1
            old_pairs = []
            new_pairs = []
            for t in range(len(token)-1):
                old_pairs.append((token[t], token[t+1]))
            for x in range(len(new_token)-1):
                new_pairs.append((new_token[x], new_token[x+1]))
            for i in old_pairs:
                pairs_count[i] -= count
                if pairs_count[i] == 0:
                    pairs_count.pop(i)
            for i in new_pairs:
                pairs_count[i] = pairs_count.get(i, 0) + count
            id_remove = set(old_pairs) - set(new_pairs)
            id_add = set(new_pairs) - set(old_pairs)
            for i in id_remove:
                pairs_dict[i].remove(j)
            for i in id_add:
                pairs_dict.setdefault(i, set()).add(j)
            token_list[j] = (tuple(new_token), count)
    return (vocab, merges)

        


        
            



        

        

            

if __name__ == "__main__":
    BPE_tokenizer("scratch/toy.txt", 270, ["<|endoftext|>"])




