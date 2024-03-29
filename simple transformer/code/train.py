import torch
from dataset import CusDataset,get_data,Collate,subseqent_mask,mask,padding_mask
from transformer import get_model
from torch.utils.data import DataLoader
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = 'cpu'
import copy
from copy import  deepcopy


eng_file_path = '../data/eng_trans.txt'
fr_file_path = '../data/french_trans.txt'
src_lang_name = 'en'
tgt_lang_name = 'fr'

train_data,val_data=get_data(eng_file_path, fr_file_path,src_lang_name, tgt_lang_name, val_frac=0.1)

SPECIAL_CHAR = {'<UNX>': 0, '<SOS>': 1, '<EOS>': 2, '<PAD>': 3}


trainDataset = CusDataset(train_data[src_lang_name], train_data[tgt_lang_name])
valDataset = CusDataset(val_data[src_lang_name], val_data[tgt_lang_name], src_lang_vocab=trainDataset.src_lang_vocab,
                        tgt_lang_vocab=trainDataset.tgt_lang_vocab)
print("########@@@@@@@@@@#############")
print(len(trainDataset.src_lang_vocab.stoi))
print(len(trainDataset.tgt_lang_vocab.stoi))
src_lang_len=len(trainDataset.src_lang_vocab.stoi)
tgt_lang_len=len(trainDataset.tgt_lang_vocab.stoi)
transformer=get_model(tgt_vocab_len=tgt_lang_len,src_vocab_len=src_lang_len,device=DEVICE)
# print(transformer)
SEQ_DIM=1

def train(model,optimizer,loss_fn,batch_size):

    total_loss = 0
    train_loader = DataLoader(trainDataset, batch_size=batch_size, num_workers=0,
                              shuffle=True, collate_fn=Collate(pad_idx=SPECIAL_CHAR['<PAD>']), pin_memory=True)

    model.train()
    for i,(src,tgt) in enumerate(train_loader):
        src=src.type(torch.IntTensor).to(device=DEVICE)
        tgt = tgt.type(torch.IntTensor).to(device=DEVICE)
        tgt_input=tgt[:,:-1]
        tgt_output=tgt[:,1:]
        tgt_mask=mask(tgt_input,SPECIAL_CHAR['<PAD>'],seq_dim=SEQ_DIM).to(device=DEVICE)
        src_mask=padding_mask(src,SPECIAL_CHAR['<PAD>']).to(device=DEVICE)

        logits=transformer(src,src_mask,tgt_input,tgt_mask)
        optimizer.zero_grad()
        loss=loss_fn(logits.reshape(-1,logits.size(-1)),tgt_output.reshape(-1).type(torch.long))
        loss.backward()
        optimizer.step()
        total_loss+=loss

        if i%50==0:
            print(f' training batch {i}/{len(train_loader)} :: loss-- {loss}')
            #break

    return total_loss/len(train_loader),model

def model_eval(model,loss_fn,batch_size):
    model.eval()
    val_loader = DataLoader(valDataset, batch_size=batch_size, num_workers=0,
                            shuffle=True, collate_fn=Collate(pad_idx=SPECIAL_CHAR['<PAD>']), pin_memory=True)
    total_val_loss = 0
    for i, (src, tgt) in enumerate(val_loader):
        src = src.type(torch.IntTensor).to(device=DEVICE)
        tgt = tgt.type(torch.IntTensor).to(device=DEVICE)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        tgt_mask = mask(tgt_input, SPECIAL_CHAR['<PAD>'], seq_dim=SEQ_DIM).to(device=DEVICE)
        src_mask = padding_mask(src, SPECIAL_CHAR['<PAD>']).to(device=DEVICE)

        logits = transformer(src, src_mask, tgt_input, tgt_mask)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1).type(torch.long))
        total_val_loss += loss
        if i%50==0:
            print(f' validation batch {i}/{len(val_loader)} :: loss-- {loss}')
            #break

    return total_val_loss/len(val_loader)





def translate(model,src_lang_sentance,vocab,start_idx,end_idx):
    model.eval()
    input_num=vocab.numericalize(src_lang_sentance)
    input_num=torch.Tensor(input_num).unsqueeze(0)
    mem=model.encode(input_num.type(torch.IntTensor).to(device=DEVICE),torch.ones_like(input_num).to(device=DEVICE))
    max_len=len(input_num)+50
    ys = torch.Tensor([[start_idx]])
    for i in range(max_len):
        ys_mask=subseqent_mask(ys.size(-1)).to(device=DEVICE)
        logits=model.decode(mem,torch.ones_like(input_num).to(device=DEVICE),ys.type(torch.long).to(device=DEVICE),ys_mask)
        next_word=torch.argmax(logits[0,-1,:],dim=-1)
        # print(next_word)
        ys=torch.cat((ys,torch.Tensor([[next_word]])),dim=-1)
        if next_word==end_idx:
            break
    ys=ys.reshape(-1).numpy()
    str_list=vocab.stringify(ys)
    str_list=str_list[1:-1]
    str=" ".join(str_list)
    print(str)
    return str


EPOCHS=2
batch_size=32
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=SPECIAL_CHAR['<PAD>'])
optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
lowest_val_score=float('inf')
best_epoch=0
patience_counter=0
max_patience_counter=1
for i in range(EPOCHS):
    train_loss=0
    # train_loss,transformer=train(transformer,optimizer,loss_fn,batch_size)
    # states_dict = deepcopy(transformer.state_dict())
    # torch.save(states_dict, "transformer_param")
    transformer = get_model(tgt_vocab_len=tgt_lang_len, src_vocab_len=src_lang_len, device=DEVICE)
    transformer.load_state_dict(torch.load("transformer_param"))
    val_loss=model_eval(transformer, loss_fn, batch_size)
    print(f' train loss ---{train_loss} ::: val loss ---{val_loss}')
    if lowest_val_score > val_loss:
        lowest_val_score=val_loss
        best_epoch=i
        # torch.save(states_dict, "best_transformer_param")
        print("best transformer_loss...."+str(val_loss))
    else:
        if(max_patience_counter==patience_counter):
            print("breaking the iteration since max patience achieved")
            break
        patience_counter+=1



# transformer.load_state_dict(torch.load("transformer_param"))
states_dict=deepcopy(transformer.state_dict())
torch.save(states_dict, "transformer_param")

#translate(transformer,"Am I right",trainDataset.tgt_lang_vocab,SPECIAL_CHAR['<SOS>'],SPECIAL_CHAR['<EOS>'])
translate(transformer,"Big people aren't always strong",trainDataset.tgt_lang_vocab,SPECIAL_CHAR['<SOS>'],SPECIAL_CHAR['<EOS>'])

translate(transformer,"Contact me for more information",trainDataset.tgt_lang_vocab,SPECIAL_CHAR['<SOS>'],SPECIAL_CHAR['<EOS>'])
translate(transformer,"Consider the following scenario",trainDataset.tgt_lang_vocab,SPECIAL_CHAR['<SOS>'],SPECIAL_CHAR['<EOS>'])


translate(transformer,"He drank three glasses of water",trainDataset.tgt_lang_vocab,SPECIAL_CHAR['<SOS>'],SPECIAL_CHAR['<EOS>'])
translate(transformer,"He will live",trainDataset.tgt_lang_vocab,SPECIAL_CHAR['<SOS>'],SPECIAL_CHAR['<EOS>'])
