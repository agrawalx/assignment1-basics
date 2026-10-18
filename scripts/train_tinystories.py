from cs336_basics import BPE_tokenizer
import pickle 
import time 
def train_tokenizer():
    start_time = time.perf_counter()
    (vocab, merges) = BPE_tokenizer("/home/yash/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt", 10000,["<|endoftext|>"]) 
    end_time = time.perf_counter()
    print(end_time - start_time)
    with open("/home/yash/assignment1-basics/outputs/trained_vocab.pkl", "wb") as file:
        pickle.dump(vocab, file)
    with open("/home/yash/assignment1-basics/outputs/trained_merges.pkl", "wb") as file:
        pickle.dump(merges, file)
    max_pair = max(vocab.items(), key=lambda x: len(x[1]))
    print(max_pair[1].decode("utf-8", errors = "replace"))
    print("saved data")
if __name__ == "__main__":
    train_tokenizer()


