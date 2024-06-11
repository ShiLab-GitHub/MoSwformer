import torch
import torch.nn as nn
from layers import VariLengthInputLayer, EncodeLayer, FeedForwardLayer, OutputLayer
from param import parameter_parser


class formerClassifier(nn.Module):
    def __init__(self, input_size, output_size, window_size=2502, d_model=64, nhead=4,
              dim_feedforward=512, num_layers=1, dropout=0.2):
        super(formerClassifier, self).__init__()

        self.window_size = window_size
        self.embedding = nn.Linear(window_size, d_model)

        # Correctly calculate sequence length after sliding window embedding
        new_seq_length = input_size - window_size + 1

        # Initialize pos_embedding
        self.pos_embedding = nn.Parameter(torch.randn(new_seq_length, d_model))

        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)

        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x):
        # Create local sliding window tokens
        windows = x.unfold(1, self.window_size, 1).contiguous()
        windows = windows.view(x.size(0), windows.size(1), -1)

        # Linear embedding
        x = self.embedding(windows)

        # Add position embeddings
        x += self.pos_embedding

        # Apply layer normalization
        x = self.layer_norm(x)

        # Encoder
        x = self.transformer_encoder(x.permute(1, 0, 2))  # Transformer requires [seq_len, batch, features]
        x = x.permute(1, 0, 2)

        # Take the output of cls_token for classification
        x = torch.mean(x, dim=1)
        # x = x[:, -1]
        x = self.fc(x)

        return x


class former(nn.Module):
    def __init__(self, input_size=1000, dropout=0., window_size=32, d_model=64, nhead=4,
              dim_feedforward=128, num_layers=1):
        super(former, self).__init__()

        self.window_size = window_size
        self.embedding = nn.Linear(window_size, d_model)

        # Correctly calculate sequence length after sliding window embedding
        new_seq_length = input_size - window_size + 1

        # Initialize cls_token and pos_embedding
        self.pos_embedding = nn.Parameter(torch.randn(new_seq_length, d_model))  # Plus 1 for cls_token

        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)

        self.layer_norm = nn.LayerNorm(d_model)


    def forward(self, x):
        # Create local sliding window tokens
        windows = x.unfold(1, self.window_size, 1).contiguous()
        windows = windows.view(x.size(0), windows.size(1), -1)

        # Linear embedding
        x = self.embedding(windows)

        # Add position embeddings
        x += self.pos_embedding

        # Apply layer normalization
        x = self.layer_norm(x)

        # Encoder
        x = self.transformer_encoder(x.permute(1, 0, 2))  # Transformer requires [seq_len, batch, features]
        x = x.permute(1, 0, 2) # [batch, seq_len, features]

        # x = x.reshape((x.shape[0], -1))
        x = torch.mean(x, dim=1)
        # x = x[:, -1]

        return x


class Multiomics_Attention_mechanism(nn.Module):
    def __init__(self):
        super().__init__()

        self.hiddim = 3
        self.globalAvgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_x1 = nn.Linear(in_features=3, out_features=self.hiddim)
        self.fc_x2 = nn.Linear(in_features=self.hiddim, out_features=3)
        self.sigmoidx = nn.Sigmoid()

    def forward(self,input_list):
        new_input_list1 = input_list[0].reshape(1, 1, input_list[0].shape[0], -1)
        new_input_list2 = input_list[1].reshape(1, 1, input_list[1].shape[0], -1)
        new_input_list3 = input_list[2].reshape(1, 1, input_list[2].shape[0], -1)
        XM = torch.cat((new_input_list1, new_input_list2, new_input_list3), 1)

        x_channel_attenttion = self.globalAvgPool(XM)

        x_channel_attenttion = x_channel_attenttion.view(x_channel_attenttion.size(0), -1)
        x_channel_attenttion = self.fc_x1(x_channel_attenttion)
        x_channel_attenttion = torch.relu(x_channel_attenttion)
        x_channel_attenttion = self.fc_x2(x_channel_attenttion)
        x_channel_attenttion = self.sigmoidx(x_channel_attenttion)
        x_channel_attenttion = x_channel_attenttion.view(x_channel_attenttion.size(0), x_channel_attenttion.size(1), 1, 1)

        XM_channel_attention = x_channel_attenttion * XM
        XM_channel_attention = torch.relu(XM_channel_attention)

        return XM_channel_attention[0]


class formerEncoder(nn.Module):
    def __init__(self, input_data_dims, hyperpm, num_class):
        super(formerEncoder, self).__init__()
        self.hyperpm = hyperpm
        self.input_data_dims = input_data_dims
        self.d_q = hyperpm.n_hidden
        self.d_k = hyperpm.n_hidden
        self.d_v = hyperpm.n_hidden
        self.n_head = hyperpm.n_head
        self.dropout = hyperpm.dropout
        self.n_layer = hyperpm.nlayer
        self.modal_num = hyperpm.nmodal
        self.n_class = num_class
        self.d_out = self.d_v * self.n_head * self.modal_num

        self.InputLayer = VariLengthInputLayer(self.input_data_dims, self.d_k, self.d_v, self.n_head, self.dropout)

        self.Encoder = []
        self.FeedForward = []

        for i in range(self.n_layer):
            encoder = EncodeLayer(self.d_k * self.n_head, self.d_k, self.d_v, self.n_head, self.dropout)
            self.add_module('encode_%d' % i, encoder)
            self.Encoder.append(encoder)

            feedforward = FeedForwardLayer(self.d_v * self.n_head, self.d_v * self.n_head, dropout=self.dropout)
            self.add_module('feed_%d' % i, feedforward)
            self.FeedForward.append(feedforward)

        d_in = self.d_v * self.n_head * self.modal_num
        self.Outputlayer = OutputLayer(d_in, self.d_v * self.n_head, self.n_class, self.modal_num, self.dropout)

    def forward(self, x):
        bs = x.size(0)
        attn_map = []
        x, _attn = self.InputLayer(x)
        attn = _attn.mean(dim=1)
        attn_map.append(attn.detach().cpu().numpy())

        for i in range(self.n_layer):
            x, _attn = self.Encoder[i](q=x, k=x, v=x, modal_num=self.modal_num)
            attn = _attn.mean(dim=1)
            x = self.FeedForward[i](x)
            attn_map.append(attn.detach().cpu().numpy())

        x = x.view(bs, -1)
        attn_embedding = attn.view(bs, -1)
        output = self.Outputlayer(x, attn_embedding)
        return output

class MOG(nn.Module):
    def __init__(self,
                 dropout=0.2,
                 d_model=64,
                 nhead=4,
                 dim_feedforward=128,
                 num_layers=1,
                 hyperpm=parameter_parser(),
                 num_class=5
                 ):
        super(MOG, self).__init__()

        self.transformer_omic1 = former(input_size=1000, dropout=dropout, window_size=1000, d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, num_layers=num_layers)
        self.transformer_omic2 = former(input_size=1000, dropout=dropout, window_size=1000, d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, num_layers=num_layers)
        self.transformer_omic3 = former(input_size=503, dropout=dropout, window_size=503, d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, num_layers=num_layers)

        self.MOAM = Multiomics_Attention_mechanism()
        self.OIRL = formerEncoder(input_data_dims=[d_model,d_model,d_model], hyperpm = hyperpm, num_class=num_class)

    def forward(self, x1, x2, x3):


        x1 = self.transformer_omic1(x1)
        x2 = self.transformer_omic2(x2)
        x3 = self.transformer_omic3(x3)

        x1 = x1.unsqueeze(0)
        x2 = x2.unsqueeze(0)
        x3 = x3.unsqueeze(0)
        x = torch.cat([x1,x2,x3], dim=0)

        atten_data_list = self.MOAM(x)
        new_x = torch.cat([atten_data_list[0],atten_data_list[1],atten_data_list[2]], dim=1)
        y = self.OIRL(new_x)

        # Logical
        return y