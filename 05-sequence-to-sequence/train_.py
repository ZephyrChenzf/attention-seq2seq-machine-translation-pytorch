import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
use_cuda = torch.cuda.is_available()

from dataset import TextDataset

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10
lang_dataset = TextDataset()
lang_dataloader = DataLoader(lang_dataset, shuffle=True)

in_len_dic = lang_dataset.input_lang_words
out_len_dic= lang_dataset.output_lang_words
emb_dim = 256
hidden_size = 256
num_epoches = 20
batch_size=1
use_attention = True


# 定义编码器
class EncoderRNN(nn.Module):
    def __init__(self, in_len_dic, emb_dim, hidden_size, num_layer=1):
        super(EncoderRNN, self).__init__()
        self.hidden_size=hidden_size
        self.num_layer=num_layer
        self.embed = nn.Embedding(in_len_dic, emb_dim)  #b,256 -> b,1,256 -> 1,b,256
        self.gru = nn.GRU(input_size=emb_dim, hidden_size=hidden_size)  # 1,b,256

    def forward(self, x, h):
        x = self.embed(x)
        x=x.unsqueeze(1)
        out = x.permute(1, 0, 2)
        for i in range(self.num_layer):
            out, h = self.gru(out, h)
        return out, h

    def initHidden(self):
        result=Variable(torch.zeros(1,1,self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

#定义普通解码器
class DecoderRNN(nn.Module):
    def __init__(self,out_len_dic,emb_dim,hidden_size,num_layer=1):
        super(DecoderRNN,self).__init__()
        self.hidden_size=hidden_size
        self.num_layer=num_layer
        self.embed=nn.Embedding(out_len_dic,emb_dim)#b,256 -> b,1,256 -> 1,b,256
        self.gru=nn.GRU(emb_dim,hidden_size,dropout=0.1)#1,b,256 ->b,256
        self.classify=nn.Linear(256,out_len_dic)

    def forward(self, x,h):
        x=self.embed(x)
        x = x.unsqueeze(1)
        out=x.permute(1,0,2)
        for i in range(self.num_layer):
            out = F.relu(out)
            out,h=self.gru(out,h)
        out=F.log_softmax(self.classify(out[0]))
        return out,h
    def initHidden(self):
        result=Variable(torch.zeros(1,1,self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

#定义带注意力集中机制的解码器
class AttentionDecoderRNN(nn.Module):
    def __init__(self,out_len_dic,emb_dim,hidden_size,num_layer=1):
        super(AttentionDecoderRNN,self).__init__()
        self.hidden_size=hidden_size
        self.num_layer=num_layer
        self.embed=nn.Embedding(out_len_dic,emb_dim)#b,1,256
        # ->1,b,256->1,b,512(与隐含层联合)->b,512
        self.attention=nn.Linear(2*hidden_size,MAX_LENGTH)#b,MaxLen
        # ->b,1,256(与注意力权重批相乘)->b,1,512(与输入联合)->b,512
        self.attention_input_conbine=nn.Linear(2*hidden_size,hidden_size)#b,256
        # ->1,b,256
        self.gru=nn.GRU(hidden_size,hidden_size,dropout=0.1)#1,b,256
        # ->b,256
        self.classify=nn.Linear(hidden_size,out_len_dic)

    def forward(self,x,h,encoder_outputs):
        embed=self.embed(x)
        #embed = embed.unsqueeze(1)
        x=embed.permute(1,0,2)
        x=torch.cat((embed,h),2)
        x=x.squeeze(0)
        atten_weights=F.softmax(self.attention(x))
        # print(atten_weights.unsqueeze(1).size())
        encoder_outputs=encoder_outputs.unsqueeze(0)
        # print(encoder_outputs.size())
        x=torch.bmm(atten_weights.unsqueeze(1),encoder_outputs)#b,1,h
        x=torch.cat((embed,x),2)
        x=x.squeeze(1)
        out=self.attention_input_conbine(x).unsqueeze(0)
        for i in range(self.num_layer):
            out = F.relu(out)
            out,h=self.gru(out,h)
        out=out[0]
        out=F.log_softmax(self.classify(out))
        return out,h,atten_weights

    def initHidden(self):
        result=Variable(torch.zeros(1,1,self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


if use_cuda:
    encoderRNN=EncoderRNN(in_len_dic,emb_dim,hidden_size).cuda()
    decoderRNN=DecoderRNN(out_len_dic,emb_dim,hidden_size,2).cuda()
    attentionDecoderRNN=AttentionDecoderRNN(out_len_dic,emb_dim,hidden_size,2).cuda()
else:
    encoderRNN=EncoderRNN(in_len_dic,emb_dim,hidden_size)
    decoderRNN=DecoderRNN(out_len_dic,emb_dim,hidden_size,2)
    attentionDecoderRNN=AttentionDecoderRNN(out_len_dic,emb_dim,hidden_size,2)

if use_attention:
    encoder=encoderRNN
    decoder=attentionDecoderRNN
else:
    encoder=encoderRNN
    decoder=decoderRNN

param=list(encoder.parameters())+list(decoder.parameters())
criterion=nn.NLLLoss()
optimzier=optim.Adam(param,lr=1e-3)

for epoch in range(num_epoches):
    train_loss=0
    train_acc=0
    for i,data in enumerate(lang_dataloader):
        x,y=data
        if use_cuda:
            x=Variable(x).cuda()
            y=Variable(y).cuda()
        else:
            x=Variable(x)
            y=Variable(y)
        encoder_outputs=Variable(torch.zeros(MAX_LENGTH,encoder.hidden_size))
        if use_cuda:
            encoder_outputs=encoder_outputs.cuda()
        encoder_hidden=encoder.initHidden()
        for ei in range(x.size(1)):    #将一句话里的每一个字的最终输出放到编码输出中
            encoder_output,encoder_hidden=encoder(x[:,ei],encoder_hidden)
            encoder_outputs[ei]=encoder_output[0][0]
        decoder_input=Variable(torch.LongTensor([[SOS_token]]))#注意转化为矩阵
        if use_cuda:
            decoder_input=decoder_input.cuda()
        decoder_hidden=encoder_hidden
        loss=0
        acc_num=0
        if use_attention:
            for di in range(y.size(1)):
                decoder_output,decoder_hidden,decoder_attention=decoder(decoder_input,decoder_hidden,encoder_outputs)
                loss+=criterion(decoder_output,y[:,di])
                topv,topi=decoder_output.data.topk(1)#取最大可能的一个,类似于使用torch.max函数
                ni=topi[0][0]
                decoder_input=Variable(torch.LongTensor([[ni]]))#前面的输出作为后面的输入
                if use_cuda:
                    decoder_input=decoder_input.cuda()
                if ni==EOS_token:
                    break
        else:
            for di in range(y.size(1)):
                decoder_output,decoder_hidden=decoder(decoder_input,decoder_hidden)
                loss+=criterion(decoder_output,y[:,di])
                topv,topi=decoder_output.data.topk(1)
                _,pre=torch.max(decoder_output,1)
                acc_num+=(pre==y[:,di]).sum().data[0]
                ni=topi[0][0]
                decoder_input=Variable(torch.LongTensor([[ni]]))
                if use_cuda:
                    decoder_input=decoder_input.cuda()
                if ni==EOS_token:
                    break
        #backward
        optimzier.zero_grad()
        loss.backward()
        optimzier.step()
        train_loss+=loss.data[0]*len(y)
        train_acc+=acc_num
        if (i + 1) % 300 == 0:
            print('[{}/{}],train loss is:{:.6f},train acc is:{:.6f}'.format(i, len(lang_dataloader),
                                                                            train_loss / (i * batch_size),
                                                                            train_acc / (i * batch_size)))
    print(
        'epoch:[{}],train loss is:{:.6f},train acc is:{:.6f}'.format(epoch,
                                                                     train_loss / (len(lang_dataloader) * batch_size),
                                                                     train_acc / (len(lang_dataloader) * batch_size)))

torch.save(encoder.state_dict(),'./model/encoder_model.pth')
torch.save(decoder.state_dict(),'./model/decoder_model.pth')