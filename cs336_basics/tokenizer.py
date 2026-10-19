import regex as re
import pickle 
import typing
from typing import (Iterable, Iterator)
class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        if special_tokens is None:
            special_tokens = []
        if special_tokens != []:
            values = set(vocab.values())
            next_token_id = len(vocab)
            for i in special_tokens:
                if i.encode("utf-8") not in values: 
                    vocab[next_token_id] = i.encode("utf-8")
                    values.add(i.encode("utf-8"))
                    next_token_id+=1
        mapping = {}
        for (key, value) in vocab.items():
            mapping[value] = key
        self.rev_index = mapping
        pos_mapping = {}
        for i in range(len(merges)):
            pos_mapping[merges[i]] = i
        self.positions = pos_mapping
        if special_tokens != []:
            sorted_special_tokens = sorted(special_tokens, key=len, reverse = True)
            pattern = re.compile("(" + "|".join(re.escape(token) for token in sorted_special_tokens) + ")")
            self.split_pattern = pattern
            self.specials = set(special_tokens)
        else:
            self.split_pattern = None
            self.specials = set()
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.pretoken_pattern = re.compile(PAT)

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens = None) -> typing.Self:
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def decode(self, ids: list[int])->str:
        text = []
        for i in ids:
            text.append(self.vocab[i])
        return b"".join(text).decode("utf-8",errors="replace")

    def encode(self, text: str)-> list[int]:
        if self.split_pattern is not None:
            parts = re.split(self.split_pattern, text)
        else:
            parts = [text]
        encoded_tokens = []
        for part in parts:
            if part in self.specials:
                encoded_tokens.append(self.rev_index[part.encode("utf-8")])
                continue
            words = re.finditer(self.pretoken_pattern, part)
            for word in words:
                word = word.group()
                tokens = [
                    bytes([b])
                    for b in word.encode("utf-8")
                ]
                while len(tokens) > 1:
                    lowest_rank = float("inf")
                    best_index = None
                    for i in range(len(tokens)-1):
                        pair = (tokens[i], tokens[i+1])
                        if (pair) in self.positions:
                            if self.positions[pair] < lowest_rank:
                                lowest_rank = self.positions[pair]
                                best_index = i 
                    if best_index is None:
                        break
                    tokens[best_index:best_index+2] = [tokens[best_index] + tokens[best_index+1]]
                ids = [
                    self.rev_index[x] for x in tokens
                ]
                encoded_tokens.extend(ids)
        return encoded_tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        buffer = ""
        for chunk in iterable:
            buffer += chunk
            if self.split_pattern is None:
                continue
            last_end = 0
            for match in self.split_pattern.finditer(buffer):
                yield from self.encode(buffer[last_end:match.end()])
                last_end = match.end()
            buffer = buffer[last_end:]
        if buffer:
            yield from self.encode(buffer)



                        
if __name__=="__main__":
    tokenizer = Tokenizer.from_files("/home/yash/assignment1-basics/outputs/trained_vocab.pkl", "/home/yash/assignment1-basics/outputs/trained_merges.pkl", ["<|endoftext|>"])
    print(tokenizer.decode([240, 159, 153, 130]))

