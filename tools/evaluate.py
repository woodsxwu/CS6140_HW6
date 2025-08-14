import torch
from sacrebleu import corpus_bleu
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import math
import os
import nltk 
import torch
import torch.nn.functional as F
import math

from bert_score import score



def detokenize(words):
    """
    Detokenize a list of words into a proper sentence
    Args:
        words: list of word tokens
    Returns:
        string with proper spacing and punctuation
    """
    if not words:
        return ""

    
    # Join words with spaces
    text = ' '.join(words)

    
    # Fix spacing around punctuation
    # Remove space before punctuation
    text = text.replace(' .', '.')
    text = text.replace(' ,', ',')
    text = text.replace(' !', '!')
    text = text.replace(' ?', '?')
    text = text.replace(' ;', ';')
    text = text.replace(' :', ':')
    text = text.replace(" '", "'")
    text = text.replace(" ' ", "'")
    
    # Fix quotes if any
    text = text.replace(' " ', '"')
    text = text.replace('" ', '"')
    return text


def greedy_decode_transformer_manual(model, src,max_len, sos_idx, eos_idx, device='cpu'):
    model.eval()
    src = src.to(device)
    batch_size, src_len = src.size()

    # 1) Create src mask once
    src_mask = (src != model.pad_idx).unsqueeze(1).unsqueeze(2)  # (batch,1,1,src_len)

    
    # 2) Encode
    enc = model.src_embedding(src)
    enc = model.pos_encoding(enc)
  
    for block in model.encoder:
        enc = block(enc, src_mask)


    # 3) Start decode with <sos>
    ys = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)  # (batch,1)

    # 4) Autoregressive loop
    for _ in range(max_len-1):
        # create combined tgt mask (pad + look-ahead)
        _, tgt_mask = model.create_masks(src, ys)  # uses your implementation

        # forward through full transformer to get logits over [batch, seq_len, V]
        logits = model(src, ys)  # (batch, current_len, V)
        # logits = torch.nn.functional.log_softmax(logits,dim=-1)
        # pick the last time-step
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (batch,1)

        # append and stop if all saw <eos>
        ys = torch.cat([ys, next_token], dim=1)
        if (next_token == eos_idx).all():
            break

    return ys.cpu().tolist()  # list of token lists

def greedy_decode_transformer(model, src, max_len, sos_idx, eos_idx, device='cpu'):
    model.eval()
    src = src.to(device)
    batch_size, _ = src.size()

    # Prepare masks
    src_key_padding_mask = (src == model.pad_idx)

    # Encode source sequence
    with torch.no_grad():
        src_emb = model.pos_encoding(model.src_embedding(src))  # (batch, src_len, d_model)
        memory = model.encoder(src_emb.transpose(0, 1), src_key_padding_mask=src_key_padding_mask)  # (src_len, batch, d_model)

        # Start with <sos> token
        ys = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)  # (batch, 1)

        for _ in range(max_len - 1):
            tgt_emb = model.pos_encoding(model.tgt_embedding(ys)).transpose(0, 1)  # (tgt_len, batch, d_model)
            tgt_mask = model.generate_square_subsequent_mask(ys.size(1)).to(device)
            tgt_key_padding_mask = (ys == model.pad_idx)

            out = model.decoder(
                tgt=tgt_emb,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask
            )

            logits = model.out_proj(out.transpose(0, 1))  # (batch, tgt_len, vocab_size)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (batch, 1)
            ys = torch.cat([ys, next_token], dim=1)

            if (next_token == eos_idx).all():
                break

    return ys[:, 1:].cpu().tolist()  # remove initial <sos> before returning

def greedy_decode_seq2seq(model,src, src_lengths, max_len, sos_idx, eos_idx, device):
    model.eval()

    # 1) move src to device
    src = src.to(device)
    # (src_lengths can stay on CPU)

    # 2) encode once
    encoder_outputs, hidden = model.encoder(src, src_lengths)

    # 3) prime decoder with <sos>
    input_token = torch.tensor([[sos_idx]],
                               dtype=torch.long,
                               device=device)

    decoded_tokens = []
    token_history = []

    for _ in range(max_len):
        # decode one step
        output, hidden = model.decoder(input_token,
                                       hidden,
                                       encoder_outputs)
        # output: (batch=1, seq_len=1, vocab)
        # squeeze out seq dim, then take the 0th batch
        logits = output.squeeze(1)[0]           # → (vocab_size,)
        probs  = F.softmax(logits, dim=-1)

        # pick top-k candidates
        k = min(5, probs.size(0))
        top_probs, top_indices = probs.topk(k)  # both are (k,)
        candidates = [idx for idx in top_indices.tolist()] 
        # loop-prevention: if last 3 tokens are all the same
        if (len(token_history) >= 1 and
            token_history[-1] == candidates[0]):
            bad = token_history[-1]
            # filter out the repeated token in Python
            candidates = [idx for idx in top_indices.tolist() if idx != bad]
            if candidates:
                next_idx = candidates[0]
                # print(f"Breaking seq2seq loop! Repeated {bad}, choosing {next_idx}")
            else:
                # fallback to second best if no alternative
                next_idx = top_indices[1].item() if k > 1 else top_indices[0].item()
        else:
            # normal greedy
            next_idx = top_indices[0].item() #after sos

        decoded_tokens.append(next_idx)
        token_history.append(next_idx)
        # if len(token_history) > max_len:
        #     token_history.pop(0)

        # stop on <eos>
        if next_idx == eos_idx:
            break

        # feed predicted token back in
        input_token = torch.tensor([[next_idx]],
                                   dtype=torch.long,
                                   device=device)

    return decoded_tokens

def calculate_scores(model, dataloader, src_vocab, tgt_vocab, device,name, max_len=20):
    """Compute BLEU and BERT score by greedy‐decoding each example (Seq2Seq or Transformer)."""
    model.eval()
    predictions, references = [], []
    # special token indices
    sos_idx = tgt_vocab.word2idx['<sos>']
    eos_idx = tgt_vocab.word2idx['<eos>']
    pad_idx = tgt_vocab.word2idx['<pad>']

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating Scores", leave=False):
            src, tgt, src_lens, tgt_lens = batch
            src = src.to(device)

            # -- Decode all examples in this batch at once --
            if name=='rnn' or name=='lstm': 
                # assume you have a seq2seq helper
                decoded_batch = []
                for i in range(src.size(0)):
                    decoded_batch.append(
                        greedy_decode_seq2seq(
                            model,
                            src[i:i+1],          # (1, src_len)
                            src_lens[i:i+1],     # (1,)
                            max_len,
                            sos_idx, eos_idx,
                            device
                        )
                    )
                # each entry is a list of token‐ids including <sos>
                decoded_batch = [ids[1:] for ids in decoded_batch]  # strip <sos>
            else:
                # Transformer: batch‐wise greedy decode
                decoded = greedy_decode_transformer(
                    model,
                    src,
                    max_len,
                    sos_idx,
                    eos_idx,
                    device
                ) #change this function to greedy_decode_transformer_manual for p4 implementation 

                # decoded shape = (batch, seq_out); drop initial <sos>
                decoded_batch = [seq[1:] for seq in decoded]
            index=0
            # -- Turn token lists into word lists and gather refs --
            for tok_list, gold_len in zip(decoded_batch, tgt_lens):
                # predicted words up to first <eos>
                pred_words = []
                for t in tok_list:
                    if t == eos_idx:
                        break
                    if t not in (pad_idx, sos_idx):
                        pred_words.append(tgt_vocab.idx2word.get(t, '<unk>'))

                # gold reference (drop <sos> and <eos>)
                
                ref_ids = tgt[index].tolist()  # you may need to index properly per example
                # here assuming tgt shape (batch, tgt_len)
                # slice out 1 : gold_len-1
                
                ref_ids = ref_ids[1:gold_len-1]
                ref_words = [tgt_vocab.idx2word.get(t, '<unk>') for t in ref_ids]
                # print(detokenize(pred_words), detokenize(ref_words))
                if pred_words and ref_words:
                    predictions.append(detokenize(pred_words))
                    references.append(detokenize(ref_words))
                    
                index+=1 
  
    # print(predictions)
   
    bleu = corpus_bleu(predictions,references)
    P, R, F1 = score(predictions, references,
                     lang='en',
                     rescale_with_baseline=True,
                     device=device) #precision,recall, f1 

    return bleu.score, F1.mean().item()  



def debug_model_output(model, dataloader, src_vocab, tgt_vocab, device, model_name, max_len=20):
    """Debug function to see what the model is actually producing"""
    model.eval()
    
    sos_idx = tgt_vocab.word2idx['<sos>']
    eos_idx = tgt_vocab.word2idx['<eos>']
    pad_idx = tgt_vocab.word2idx['<pad>']
    
    print("=== DEBUGGING MODEL OUTPUT ===")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 1:  # Only check first batch
                break
                
            src, tgt, src_lengths, tgt_lengths = batch 
            src = src.to(device)
            # print(src)
            
            for i in range(min(3, src.size(0))):  # Check first 3 examples
                print(f"\n--- Example {i+1} ---")
                
                # Show source
                src_tokens = src[i, :src_lengths[i]].cpu().tolist()
                print(src_tokens)
                src_words = [src_vocab.idx2word.get(tok, f'UNK_{tok}') for tok in src_tokens]
                # print(src_words)
                print(f"Source: {detokenize(src_words)}")
                
                # Show target
                
                tgt_tokens = tgt[i, 1:tgt_lengths[i]-1].cpu().tolist()  # Skip <sos>/<eos>
                print(tgt_tokens)
                tgt_words = [tgt_vocab.idx2word.get(tok, f'UNK_{tok}') for tok in tgt_tokens]
                print(f"Target: {detokenize(tgt_words)}")
                
                # Show model prediction
                src_sentence = src[i:i+1]
                src_len = src_lengths[i:i+1]
                print(src_sentence)
                # print(sos_idx,eos_idx)
                try:
                    if model_name=='rnn' or model_name=='lstm':  # Seq2Seq
                        decoded = greedy_decode_seq2seq(
                            model, src_sentence, src_len, max_len,
                            sos_idx, eos_idx, device
                        )
                    else:  # Transformer
                        output = greedy_decode_transformer(
                            model, src_sentence, max_len,
                            sos_idx, eos_idx, device
                        ) #change this function to greedy_decode_transformer_manual for p4 implementation 

                        # print(output)
                        decoded = output[0][1:]#[seq[1:] for seq in output]
                        # print(decoded)
                    pred_words = []
                    for token in decoded:
                        if token == eos_idx:
                            break
                        if token != pad_idx and token != sos_idx:
                            pred_words.append(tgt_vocab.idx2word.get(token, f'UNK_{token}'))
                    
                    print(f"Prediction: {detokenize(pred_words)}")
                    print(f"Raw decoded: {decoded}")  # First 10 tokens
                    
                except Exception as e:
                    print(f"Error during decoding: {e}")
                    import traceback
                    traceback.print_exc()

