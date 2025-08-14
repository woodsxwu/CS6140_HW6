import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re
import nltk
# nltk.download('punkt_tab') if you dont have this uncomment 

class Vocabulary:
    def __init__(self):
        self.word2idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        self.idx2word = {0: '<pad>', 1: '<sos>', 2: '<eos>', 3: '<unk>'}
        self.word_count = Counter()
        
    def build_vocab(self, sentences: list[str], min_freq:int=2) -> None:
        """
        Build vocabulary from sentences
        Args:
            sentences: list of tokenized sentences
            min_freq: minimum frequency for a word to be included
        """
        # Count word frequencies
        for sentence in sentences:
            for word in sentence:
                # print(word)
                self.word_count[word] += 1
        
        # Add words with frequency >= min_freq to vocabulary
        idx = len(self.word2idx)
        for word, count in self.word_count.items():
            if count >= min_freq and word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
    
    def encode(self, sentence: list[str]) -> list[int]:
        """
        Convert sentence to indices
        Args:
            sentence: list of tokens
        Returns:
            list of indices
        """
        return [self.word2idx.get(word, self.word2idx['<unk>']) for word in sentence]
    
    def decode(self, indices: list[int]) -> list[str]:
        """
        Convert indices back to sentence
        Args:
            indices: list of indices
        Returns:
            list of tokens
        """
        return [self.idx2word.get(idx, '<unk>') for idx in indices]
    
    def __len__(self):
        return len(self.word2idx)

class TranslationDataset(Dataset):
    def __init__(self, src_sentences: list[str], tgt_sentences: list[str], src_vocab: Vocabulary, tgt_vocab: Vocabulary):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        
    def __len__(self):
        return len(self.src_sentences)
        
    def __getitem__(self, idx):
        src = self.src_vocab.encode(self.src_sentences[idx])
        tgt = self.tgt_vocab.encode(self.tgt_sentences[idx])
        
        # Add <sos> and <eos> tokens to target
        tgt = [self.tgt_vocab.word2idx['<sos>']] + tgt + [self.tgt_vocab.word2idx['<eos>']]
        
        return torch.tensor(src), torch.tensor(tgt)
    
def better_tokenize(text: str) -> list[str]:
    """
    Use a more sophisticated tokenizer
    """
      
    return nltk.word_tokenize(text.lower())
    
    #or spacy 

def simple_tokenize(text: str) -> list[str]:
    """Simple but effective tokenizer"""
    import re
    text = text.lower().strip()
    # Keep contractions together
    text = re.sub(r"'", " '", text)
    # Separate punctuation only when needed
    text = re.sub(r"([.!?,])", r" \1", text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    return text.split()

def load_data(filename: str, max_len:int=20): 
    """
    Load parallel sentences from file
    Args:
        filename: path to data file
        max_len: maximum sentence length
    Returns:
        src_sentences, tgt_sentences: lists of tokenized sentences
    """
    src_sentences = []
    tgt_sentences = []
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if '\t' in line:
                # print(line.strip().split('\t'))
                src, tgt = line.strip().split('\t')[1],line.strip().split('\t')[3]
                src_tokens =  simple_tokenize(src) 
                tgt_tokens = simple_tokenize(tgt) 
                
                if len(src_tokens) <= max_len and len(tgt_tokens) <= max_len:
                    src_sentences.append(src_tokens)
                    tgt_sentences.append(tgt_tokens)
                    
    return src_sentences, tgt_sentences

def collate_fn(batch):
    """
    Collate function for DataLoader
    Pads sequences to same length in batch
    """
    # Separate source and target sequences
    src_batch, tgt_batch = zip(*batch)
    
    # Find max lengths in this batch
    src_max_len = max(len(s) for s in src_batch)
    tgt_max_len = max(len(t) for t in tgt_batch)
    
    # Initialize lists for padded sequences and lengths
    src_padded = []
    tgt_padded = []
    src_lengths = []
    tgt_lengths = []
    
    # Pad each sequence in the batch
    for src, tgt in zip(src_batch, tgt_batch):
        src_len = len(src)
        tgt_len = len(tgt)
        # Store actual lengths
        src_lengths.append(src_len)
        tgt_lengths.append(tgt_len)

        # Pad source sequence
        src_padding = torch.zeros(src_max_len - src_len, dtype=torch.long)
        src_padded.append(torch.cat([src, src_padding]))
        
        # Pad target sequence
        tgt_padding = torch.zeros(tgt_max_len - tgt_len, dtype=torch.long)
        tgt_padded.append(torch.cat([tgt, tgt_padding]))
        
        
    
    # Stack into tensors
    src_padded = torch.stack(src_padded)
    tgt_padded = torch.stack(tgt_padded)
    src_lengths = torch.tensor(src_lengths)
    tgt_lengths = torch.tensor(tgt_lengths)
    
    return src_padded, tgt_padded, src_lengths, tgt_lengths

