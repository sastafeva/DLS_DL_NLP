import random
import torch
from torch import nn
from torch.nn import functional as F


def softmax(x, temperature=3): # use your temperature
    e_x = torch.exp(x / temperature)
    return e_x / torch.sum(e_x, dim=0)
    

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout, bidirectional):
        super().__init__()
        
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers, bidirectional=bidirectional)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, src):
        
        #src = [src sent len, batch size]
        
        embedded = self.dropout(self.embedding(src))
        #embedded = [src sent len, batch size, emb dim]
        
        outputs, (hidden, cell) = self.rnn(embedded)
        
        #outputs = [src sent len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #outputs are always from the top hidden layer
        #hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        if self.bidirectional:
            hidden = hidden.reshape(self.n_layers, 2, -1, self.hid_dim)
            hidden = hidden.transpose(1, 2).reshape(self.n_layers, -1, 2 * self.hid_dim)

            cell = cell.reshape(self.n_layers, 2, -1, self.hid_dim)
            cell = cell.transpose(1, 2).reshape(self.n_layers, -1, 2 * self.hid_dim)

        return outputs, hidden[-1, :, :], cell[-1, :, :]


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim, bidirectional):
        super().__init__()
        
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.bidirectional = bidirectional
        
        self.attn = nn.Linear((enc_hid_dim) * (1 + bidirectional) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1)
        
    def forward(self, hidden, encoder_outputs):
        
        # encoder_outputs = [src sent len, batch size, (enc_hid_dim) * (1 + bidirectional)]
        # hidden = [batch size, (enc_hid_dim) * (1 + bidirectional)]
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        # repeat hidden and concatenate it with encoder_outputs
        hidden = hidden.repeat(src_len, 1, 1)
        # encoder_outputs = encoder_outputs.permute(1, 0, 2)
        cat_hidden_out = torch.cat((hidden, encoder_outputs), dim = 2)
        
        # calculate energy
        energy = torch.tanh(self.attn(cat_hidden_out))
        #energy = [batch size, src len, dec hid dim]

        # get attention, use softmax function which is defined, can change temperature
        attention = self.v(energy)
        attention = softmax(attention)
            
        return attention
    
    
class DecoderWithAttention(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)        
        self.rnn = nn.GRU(dec_hid_dim + emb_dim, dec_hid_dim) # use GRU        
        self.out = nn.Linear(dec_hid_dim * 2 + emb_dim, output_dim) # linear layer to get next word        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):
        #input = [batch size]
        #hidden = [batch size, dec_hid_dim]
                
        input = input.unsqueeze(0) # because only one word, no words sequence         
        #input = [1, batch size]
        
        embedded = self.dropout(self.embedding(input))        
        #embedded = [1, batch size, emb dim]
        
        # get weighted sum of encoder_outputs
        a = self.attention(hidden, encoder_outputs)
        #a = [batch size, src len, 1]
               
        # a * encoder_outputs = [src len, batch size, dec_hid_dim]
        weighted = torch.sum(a * encoder_outputs, dim=0, keepdim=True)
        #weighted = [1, batch size, dec_hid_dim]

        # concatenate weighted sum and embedded, break through the GRU
        cat_weighted = torch.cat((weighted, embedded), dim = 2)
        #cat_weighted = [1, batch size, dec_hid_dim + emb dim]
            
        output, hidden = self.rnn(cat_weighted, hidden.unsqueeze(0))
        #output = [1, batch size, dec hid dim]
        #hidden = [1, batch size, dec hid dim]
        assert (output == hidden).all()

        # get predictions 
        prediction = self.out(torch.cat((embedded, weighted, hidden), dim = 2))
        #prediction = [1, batch size, output dim]
        
        return prediction, hidden.squeeze(0)
        

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        if encoder.bidirectional:
            assert encoder.hid_dim * 2 == decoder.dec_hid_dim, \
                "Hidden dimensions of encoder and decoder must be equal!"
        else:
            assert encoder.hid_dim == decoder.dec_hid_dim, \
                    "Hidden dimensions of encoder and decoder must be equal!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        # src = [src sent len, batch size]
        # trg = [trg sent len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        # Again, now batch is the first dimention instead of zero
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        enc_states, hidden, cell = self.encoder(src)
        
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        for t in range(1, trg_len):

            output, hidden = self.decoder(input, hidden, enc_states)

            outputs[t] = output
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            #get the highest predicted token from our predictions
            top1 = output.argmax(-1).flatten()
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1
        
        return outputs