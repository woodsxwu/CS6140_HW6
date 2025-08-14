import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence


class CustomRNNCell(nn.Module):
    """
    Custom RNN cell implementation from scratch
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize weight matrices and bias vectors
        self.W_ih = nn.Parameter(torch.randn(input_size, hidden_size))
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_hh = nn.Parameter(torch.zeros(hidden_size))

        # Initialize weights using Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.W_ih)
        nn.init.xavier_uniform_(self.W_hh)
        nn.init.zeros_(self.b_hh)

    def forward(self, input: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for one time step
        Args:
            input: input at current time step [batch_size, input_size]
            hidden: hidden state from previous time step [batch_size, hidden_size]
        Returns:
            new_hidden: updated hidden state [batch_size, hidden_size]
        """
        # RNN cell forward pass: h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b_hh)
        input_transform = torch.mm(input, self.W_ih)
        hidden_transform = torch.mm(hidden, self.W_hh) + self.b_hh
        new_hidden = torch.tanh(input_transform + hidden_transform)
        return new_hidden


class LSTMCell(nn.Module):
    """
    LSTM cell implementation from scratch
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize weight matrices and bias vectors for LSTM gates
        # Input gate
        self.W_ii = nn.Parameter(torch.randn(input_size, hidden_size))
        self.W_hi = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.zeros(hidden_size))

        # Forget gate
        self.W_if = nn.Parameter(torch.randn(input_size, hidden_size))
        self.W_hf = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.zeros(hidden_size))

        # Input node gate (candidate values)
        self.W_in = nn.Parameter(torch.randn(input_size, hidden_size))
        self.W_hn = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_n = nn.Parameter(torch.zeros(hidden_size))

        # Output gate
        self.W_io = nn.Parameter(torch.randn(input_size, hidden_size))
        self.W_ho = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.zeros(hidden_size))

        # Initialize weights with xavier_uniform_, biases with zeros_
        for param in [self.W_ii, self.W_hi, self.W_if, self.W_hf,
                      self.W_in, self.W_hn, self.W_io, self.W_ho]:
            nn.init.xavier_uniform_(param)

        for param in [self.b_i, self.b_f, self.b_n, self.b_o]:
            nn.init.zeros_(param)

    def forward(self, input: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor]) -> tuple[
        torch.Tensor, torch.Tensor]:
        """
        Forward pass for one time step
        Args:
            input: input at current time step [batch_size, input_size]
            states: tuple of (hidden_state, cell_state) from previous time step
                    both with shape [batch_size, hidden_size]
        Returns:
            new_hidden: updated hidden state [batch_size, hidden_size]
            new_cell: updated cell state [batch_size, hidden_size]
        """
        hidden, cell = states

        # LSTM gates computation
        # Forget gate
        f_t = torch.sigmoid(torch.mm(input, self.W_if) + torch.mm(hidden, self.W_hf) + self.b_f)

        # Input gate
        i_t = torch.sigmoid(torch.mm(input, self.W_ii) + torch.mm(hidden, self.W_hi) + self.b_i)

        # Input node values (candidate)
        n_t = torch.tanh(torch.mm(input, self.W_in) + torch.mm(hidden, self.W_hn) + self.b_n)

        # Output gate
        o_t = torch.sigmoid(torch.mm(input, self.W_io) + torch.mm(hidden, self.W_ho) + self.b_o)

        # Update cell state
        new_cell = f_t * cell + i_t * n_t

        # Update hidden state
        new_hidden = o_t * torch.tanh(new_cell)

        return new_hidden, new_cell


class CustomRNN(nn.Module):
    """
    Multi-layer Custom RNN implementation using custom RNN cells,
    with support for PackedSequence inputs.
    """

    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int = 1, batch_first: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        # build one cell per layer
        self.cells = nn.ModuleList([
            CustomRNNCell(input_size if l == 0 else hidden_size, hidden_size)
            for l in range(num_layers)
        ])

    def forward(self, input, hidden: torch.Tensor = None):
        # detect PackedSequence
        is_packed = isinstance(input, PackedSequence)
        if is_packed:
            # unpack -> padded [B, T, D] + lengths
            padded, lengths = pad_packed_sequence(
                input, batch_first=self.batch_first
            )
            outs, h_n = self._forward_unpacked(padded, hidden)
            # re-pack before returning
            packed_out = pack_padded_sequence(
                outs, lengths,
                batch_first=self.batch_first,
                enforce_sorted=False
            )
            return packed_out, h_n
        else:
            return self._forward_unpacked(input, hidden)

    def _forward_unpacked(self, input: torch.Tensor, hidden: torch.Tensor = None):
        # bring to [B, T, D] if needed
        if not self.batch_first:
            input = input.transpose(0, 1)
        B, T, _ = input.size()

        # init or unpack hidden into list of [B, H]
        if hidden is None:
            h_t = [input.new_zeros(B, self.hidden_size)
                   for _ in range(self.num_layers)]
        else:
            # hidden: [L, B, H]
            h_t = [hidden[layer] for layer in range(self.num_layers)]

        # collect all time‑step outputs
        outputs = input.new_zeros(B, T, self.hidden_size)
        for t in range(T):
            x = input[:, t, :]
            for l, cell in enumerate(self.cells):
                h = cell(x, h_t[l])  # your CustomRNNCell returns next hidden
                h_t[l] = h
                x = h
            outputs[:, t, :] = x

        # stack back into [L, B, H]
        h_n = torch.stack(h_t, dim=0)

        # if originally seq_first, transpose back
        if not self.batch_first:
            outputs = outputs.transpose(0, 1)

        return outputs, h_n


class CustomLSTM(nn.Module):
    """
    Multi-layer LSTM implementation using custom LSTM cells
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, batch_first: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            in_size = input_size if layer == 0 else hidden_size
            self.cells.append(LSTMCell(in_size, hidden_size))

    def forward(self, input, states=None):
        # 1) if PackedSequence, unpack to a Tensor + lengths
        is_packed = isinstance(input, PackedSequence)
        if is_packed:
            padded, lengths = pad_packed_sequence(input, batch_first=self.batch_first)
            outputs, (h_n, c_n) = self._forward_unpacked(padded, states)
            packed_out = pack_padded_sequence(
                outputs, lengths,
                batch_first=self.batch_first,
                enforce_sorted=False
            )
            return packed_out, (h_n, c_n)
        else:
            return self._forward_unpacked(input, states)

    def _forward_unpacked(self, input: torch.Tensor, states):
        # -- exactly your old code, but expecting a plain Tensor:
        #    [B, T, D_in] if batch_first else you'd have transposed already
        if not self.batch_first:
            input = input.transpose(0, 1)  # bring to [B, T, D_in]
        B, T, _ = input.size()

        # init or unpack states
        if states is None:
            h_t = [input.new_zeros(B, self.hidden_size) for _ in range(self.num_layers)]
            c_t = [input.new_zeros(B, self.hidden_size) for _ in range(self.num_layers)]
        else:
            h0, c0 = states  # each [L, B, H]
            h_t = [h0[layer] for layer in range(self.num_layers)]
            c_t = [c0[layer] for layer in range(self.num_layers)]

        outputs = input.new_zeros(B, T, self.hidden_size)
        for t in range(T):
            x = input[:, t, :]
            for layer_idx, cell in enumerate(self.cells):
                h, c = cell(x, (h_t[layer_idx], c_t[layer_idx]))
                h_t[layer_idx], c_t[layer_idx] = h, c
                x = h
            outputs[:, t, :] = x

        h_n = torch.stack(h_t, dim=0)  # [L, B, H]
        c_n = torch.stack(c_t, dim=0)

        if not self.batch_first:
            outputs = outputs.transpose(0, 1)  # back to [T, B, H]

        return outputs, (h_n, c_n)


class Encoder(nn.Module):
    def __init__(self, input_size: int, embed_size: int, hidden_size: int, num_layers: int = 1, rnn_type: str = 'lstm'):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type

        # Initialize embedding layer
        self.embedding = nn.Embedding(input_size, embed_size)

        # Initialize RNN (CustomRNN) or LSTM (CustomLSTM) based on rnn_type
        if rnn_type == 'lstm':
            self.rnn = CustomLSTM(embed_size, hidden_size, num_layers, batch_first=True)
        else:  # rnn_type == 'rnn'
            self.rnn = CustomRNN(embed_size, hidden_size, num_layers, batch_first=True)

    def forward(self, src, src_lengths) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            src: source sequences [batch_size, seq_len]
            src_lengths: actual lengths of sequences [batch_size]
        Returns:
            outputs: all hidden states [batch_size, seq_len, hidden_size]
            hidden: final hidden state
        """
        # Embed input sequences
        embedded = self.embedding(src)  # [batch_size, seq_len, embed_size]

        # Pack padded sequence
        packed = pack_padded_sequence(embedded, src_lengths, batch_first=True, enforce_sorted=False)

        # Pass through RNN/LSTM
        packed_outputs, hidden = self.rnn(packed)

        # Unpack sequence
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)

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
        if rnn_type == 'lstm':
            self.rnn = CustomLSTM(embed_size, hidden_size, num_layers, batch_first=True)
        else:  # rnn_type == 'rnn'
            self.rnn = CustomRNN(embed_size, hidden_size, num_layers, batch_first=True)

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
        # Embed input token
        embedded = self.embedding(input)  # [batch_size, 1, embed_size]

        # Pass through RNN/LSTM
        rnn_output, hidden = self.rnn(embedded, hidden)

        # Apply output projection (squeeze the sequence dimension)
        output = self.out(rnn_output.squeeze(1))  # [batch_size, output_size]

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