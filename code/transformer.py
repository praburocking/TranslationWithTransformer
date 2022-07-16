

import os
from os.path import exists
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import math
import copy

import warnings
from torch.nn import LayerNorm
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EncoderDecoder(nn.Module):
  """
  first encoderDecoder (standard one)
  """
  def __init__(self,encoder,decoder,src_vocab_len,tgt_vocab_len,d_model,generator):
    super(EncoderDecoder,self).__init__()
    
    self.encoder=encoder
    self.decoder=decoder
    self.src_embed=nn.Embedding(src_vocab_len,d_model)
    self.tgt_embed=nn.Embedding(tgt_vocab_len,d_model)
    self.pos=PositionalEncoding(d_model)
    self.gen=generator
    
  def forward(self,src,src_mask,tgt,tgt_mask):
    return self.decode(self.encode(src,src_mask),src_mask,tgt,tgt_mask)

  def encode(self,src,mask_src):
    # breakpoint()
    return self.encoder(self.pos(self.src_embed(src)),mask_src)

  def decode(self,mem,mem_mask,tgt,tgt_mask):
    # breakpoint()
    return self.gen(self.decoder(mem,mem_mask,self.pos(self.tgt_embed(tgt)),tgt_mask))

class Generator(nn.Module):
  def __init__(self,d_model,vocab_size):
    super(Generator,self).__init__()
    self.linear=nn.Linear(d_model,vocab_size)
    
  def forward(self,x):
    return log_softmax(self.linear(x),dim=-1)

def clone(module,N):
  return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
  def __init__(self,N,module):
    super(Encoder,self).__init__()
    self.encodeLayers=clone(module,N)
      
  def forward(self,x,mask):
    for encodeLayer in self.encodeLayers:
      x=encodeLayer(x,mask)
    return x

class Decoder(nn.Module):
  def __init__(self,N,module):
    super(Decoder,self).__init__()
    self.decodeLayers=clone(module,N)
    
  def forward(self,mem,mem_mask,y,y_mask,):
    for decodeLayer in self.decodeLayers:
      x=decodeLayer(mem,mem_mask,y,y_mask)
    return x

class SubLayerConnection(nn.Module):
  def __init__(self,embed_size,dropout=0.01):
    super(SubLayerConnection,self).__init__()
    self.norm = LayerNorm(embed_size)
    self.dropout = nn.Dropout(dropout)
    
  def forward(self,subModule,x):
    return x+self.dropout(self.norm(subModule(x)))

class FeedForward(nn.Module):
  def __init__(self,size):
    super(FeedForward,self).__init__()
    self.ff1=nn.Linear(size,size)
    self.ff2=nn.Linear(size,size)
    self.relu=nn.ReLU()
    
  def forward(self,x):
    x=self.relu(self.ff1(x))
    return self.ff2(x)

class EncodeLayer(nn.Module):
  def __init__(self,head,d_model):
    super(EncodeLayer,self).__init__()
    self.multiHeadAtten= MultiHeadAttention(head,d_model)
    self.feedForward=FeedForward(d_model)
    self.subLayerCon=clone(SubLayerConnection(d_model),2)
    

  def forward(self,x,x_mask):
    x=self.subLayerCon[0](lambda x: self.multiHeadAtten(x,x,x,x_mask),x)
    x=self.subLayerCon[1](self.feedForward,x)
    return x

class DecodeLayer(nn.Module):
  def __init__(self,head,d_model):
    super(DecodeLayer,self).__init__()
    self.multiHeadAtten1=MultiHeadAttention(head,d_model)
    self.multiHeadAtten2=MultiHeadAttention(head,d_model)
    self.feedForward=FeedForward(d_model)
    self.subLayerCon=clone(SubLayerConnection(d_model),3)
    

  def forward(self,mem,mem_mask,tgt,tgt_mask):
    # print("***********decoder")
   
    x=self.subLayerCon[0](lambda tgt: self.multiHeadAtten1(tgt,tgt,tgt,tgt_mask),tgt)
    # print("decoder::: MH---1 size of x  "+str(x.size())+"size of mem"+str(mem.size()))
    x=self.subLayerCon[1](lambda x: self.multiHeadAtten2(x,mem,mem,mem_mask),x)
    # print("decoder::: MH---2 size of x  "+str(x.size())+"size of mem"+str(mem.size()))

    x=self.subLayerCon[2](self.feedForward,x)
    # print("decoder::: FF---1 size of x  "+str(x.size())+"size of mem"+str(mem.size()))
    return x

class Embedding(nn.Module):
  def __init__(self,vocab_len,d_model):
    super(Embedding,self).__init__()
    self.embedding=nn.Embedding(vocab_len,d_model)
  def forward(self,x):
    return self.embedding(x)


class Attention(nn.Module):
  def __init__(self,embed_size,d_k_size,d_v_size):
    super(Attention,self).__init__()
    self.k_lin=nn.Linear(embed_size,d_k_size)
    self.q_lin=nn.Linear(embed_size,d_k_size)
    self.v_lin=nn.Linear(embed_size,d_v_size)
    self.d_k_size=d_k_size
    self.d_v_size=d_v_size
    
  def forward(self,query,key,value,mask=None):
    # breakpoint()
    key=self.k_lin(key)#batch,seq,d_k
    query=self.q_lin(query)#batch,seq,d_k
    value=self.v_lin(value)#batch,seq,d_v
    #breakpoint()
    attent=torch.matmul(query,key.transpose(-1,-2))/math.sqrt(self.d_k_size)#batch,seq(query),seq(value)
    if mask is not None:
      # breakpoint()
      attent.masked_fill_(mask,float('-inf'))
    attent=attent.softmax(dim=-1)
    return attent, attent@value
    
class MultiHeadAttention(nn.Module):
  def __init__(self,h,d_model,dropout=0.1):
    super(MultiHeadAttention,self).__init__()
    assert d_model%h==0
    self.d_k=d_model//h
    self.d_v=d_model//h
    self.d_model=d_model
    self.attHeads=clone(Attention(d_model,self.d_k,self.d_v),h)

  def forward(self,query,key,value,x_mask):
    appendVal=torch.Tensor([])
    
    for i,attHead in enumerate(self.attHeads):
      # breakpoint()
      _,attentVal=attHead(query,key,value,x_mask)
      appendVal=torch.cat((appendVal,attentVal),-1)
    return appendVal

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout=0.01, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        print(div_term)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        print(div_term)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)



def get_model(tgt_vocab_len=15000,src_vocab_len=15000,num_head=8,num_layer=6,d_model=512,device=DEVICE):
    enc = Encoder(num_layer, copy.deepcopy(EncodeLayer(num_head, d_model)))
    dec = Decoder(num_layer, copy.deepcopy(DecodeLayer(num_head, d_model)))
    encDec = EncoderDecoder(enc, dec, src_vocab_len, tgt_vocab_len, d_model, Generator(d_model, tgt_vocab_len))
    for p in encDec.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    transformer = encDec.to(device)
    return transformer


