import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class Encoder(nn.Module):
    def __init__(self, input_size: int, embed_size: int, hidden_size: int, num_layers: int = 1, rnn_type: str = 'lstm'):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type

        # Initialize embedding layer
        self.embedding = nn.Embedding(input_size, embed_size)

        # Initialize RNN (nn.RNN) or LSTM (nn.LSTM) based on rnn_type
        # Hint: Use batch_first=True for easier handling
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        else:  # rnn_type == 'rnn'
            self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True)

    def forward(self, src, src_lengths) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            src: source sequences [batch_size, seq_len]
            src_lengths: actual lengths of sequences [batch_size]
        Returns:
            outputs: all hidden states [batch_size, seq_len, hidden_size]
            hidden: final hidden state
        """
        # 1. Embed input sequences
        embedded = self.embedding(src)

        # 2. Pack padded sequence (nn.utils.rnn.pack_padded_sequence)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False)

        # 3. Pass through RNN/LSTM
        packed_outputs, hidden = self.rnn(packed)

        # 4. Unpack sequence (nn.utils.rnn.pad_packed_sequence)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)

        # 5. Return outputs and hidden state
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, output_size: int, embed_size: int, hidden_size: int, num_layers: int = 1,
                 rnn_type: str = 'lstm'):
        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type

        # Initialize embedding layer
        self.embedding = nn.Embedding(output_size, embed_size)

        # Initialize RNN or LSTM
        # Note: Input size is embed_size, not hidden_size
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        else:  # rnn_type == 'rnn'
            self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True)

        # Initialize output projection layer: from hidden_size to output_size
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input: torch.Tensor, hidden: torch.Tensor, encoder_outputs: torch.Tensor = None) -> tuple[
        torch.Tensor, torch.Tensor]:
        """
        Decode one time step
        Args:
            input: input token [batch_size, 1]
            hidden: hidden state from previous time step
            encoder_outputs: encoder outputs (for attention, optional)
        Returns:
            output: predictions [batch_size, output_size]
            hidden: updated hidden state
        """
        # 1. Embed input token
        embedded = self.embedding(input)

        # 2. Pass through RNN/LSTM, including the hidden state from the previous time step
        rnn_output, hidden = self.rnn(embedded, hidden)

        # 3. Apply output projection (need to squeeze the rnn output dim=1)
        output = self.out(rnn_output.squeeze(1))

        # 4. Return output and hidden state
        return output, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        # Ensure hidden dimensions match
        assert encoder.hidden_size == decoder.hidden_size
        assert encoder.num_layers == decoder.num_layers

    def forward(self, src: torch.Tensor, src_lengths: torch.Tensor, tgt: torch.Tensor,
                teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        """
        Forward pass with teacher forcing
        Args:
            src: source sequences [batch_size, src_len]
            src_lengths: actual lengths of source sequences
            tgt: target sequences [batch_size, tgt_len]
            teacher_forcing_ratio: probability of using teacher forcing
        Returns:
            outputs: predictions [batch_size, tgt_len, output_size]
        """
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        tgt_vocab_size = self.decoder.output_size

        # Initialize tensor to store outputs
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(src.device)

        # Encode source sequence
        encoder_outputs, hidden = self.encoder(src, src_lengths)

        # First input to decoder is <sos> token (already in tgt[:, 0]), should always be shape [batch_size, 1]
        input = tgt[:, 0].unsqueeze(1)

        # Decode sequence step by step
        for t in range(1, tgt_len):
            #     1. Pass through decoder, get predictions and next hidden state
            decoder_output, hidden = self.decoder(input, hidden, encoder_outputs)
            #     2. Store predictions
            outputs[:, t] = decoder_output
            #     3. Decide whether to use teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            #     4. Get next input (either from target if teacher forcing or highest predicted token) should be shape [batch_size, 1]
            if teacher_force:
                input = tgt[:, t].unsqueeze(1)
            else:
                input = torch.argmax(decoder_output, dim=1).unsqueeze(1)

        return outputs


def create_rnn_model(src_vocab_size: int, tgt_vocab_size: int, embed_size: int = 256,
                     hidden_size: int = 512, num_layers: int = 2):
    """
    Create RNN-based seq2seq model
    Args:
        src_vocab_size: source vocabulary size
        tgt_vocab_size: target vocabulary size
        embed_size: embedding dimension
        hidden_size: hidden state dimension
        num_layers: number of RNN layers
    Returns:
        Seq2Seq model
    """
    # Create encoder and decoder with 'rnn' type
    encoder = Encoder(input_size=src_vocab_size, embed_size=embed_size, hidden_size=hidden_size, num_layers=num_layers,
                      rnn_type="rnn")
    decoder = Decoder(output_size=tgt_vocab_size, embed_size=embed_size, hidden_size=hidden_size, num_layers=num_layers,
                      rnn_type="rnn")
    return Seq2Seq(encoder, decoder)


def create_lstm_model(src_vocab_size: int, tgt_vocab_size: int, embed_size: int = 256,
                      hidden_size: int = 512, num_layers: int = 2):
    """
    Create LSTM-based seq2seq model
    Args:
        src_vocab_size: source vocabulary size
        tgt_vocab_size: target vocabulary size
        embed_size: embedding dimension
        hidden_size: hidden state dimension
        num_layers: number of LSTM layers
    Returns:
        Seq2Seq model
    """
    # Create encoder and decoder with 'lstm' type
    encoder = Encoder(input_size=src_vocab_size, embed_size=embed_size, hidden_size=hidden_size, num_layers=num_layers,
                      rnn_type="lstm")
    decoder = Decoder(output_size=tgt_vocab_size, embed_size=embed_size, hidden_size=hidden_size, num_layers=num_layers,
                      rnn_type="lstm")
    return Seq2Seq(encoder, decoder)