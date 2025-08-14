import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, d_ff=2048,
                 num_layers=6, dropout=0.1, max_len=5000, pad_idx=0):
        super(Transformer, self).__init__()
        self.pad_idx = pad_idx

        # Initialize source and target embedding layers using nn.Embedding
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # Initialize positional encoding using the PositionalEncoding class
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        # Create a TransformerEncoderLayer using nn.TransformerEncoderLayer
        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)

        # Use nn.TransformerEncoder to create the encoder stack with multiple layers name it "encoder"
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # Create a TransformerDecoderLayer using nn.TransformerDecoderLayer
        decoder_layer = nn.TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)

        # Use nn.TransformerDecoder to create the decoder stack with multiple layers name it "decoder"
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        # Create a final linear layer (output projection) that maps decoder output to target vocab size
        self.out_proj = nn.Linear(d_model, tgt_vocab_size)

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

    def forward(self, src, tgt):
        src_key_padding_mask = (src == self.pad_idx)
        tgt_key_padding_mask = (tgt == self.pad_idx)
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)

        src_emb = self.pos_encoding(self.src_embedding(src))
        tgt_emb = self.pos_encoding(self.tgt_embedding(tgt))

        memory = self.encoder(src_emb.transpose(0, 1), src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt_emb.transpose(0, 1), memory,
                              tgt_mask=tgt_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=src_key_padding_mask)

        return self.out_proj(output.transpose(0, 1))