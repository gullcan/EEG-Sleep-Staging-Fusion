# 参考自(reference) https://github.com/flower-kyo/Tinysleepnet-pytorch/blob/main/network.py
# 网络上增加了双向RNN配置，并测试使用了GRU和Attention
# 因为重写了数据集加载的代码，代码更加简洁，可灵活调整输入序列长度seq_len

import torch
import torch.nn as nn
from collections import OrderedDict

class SleepNet(nn.Module):
    def __init__(self, hidden_size=128, seq_len=20, is_bidirectional=False, network="GRU"):
        super(SleepNet, self).__init__()
        self.padding_edf = {
            'conv1': (22, 22),
            'max_pool1': (2, 2),
            'conv2': (3, 4),
            'max_pool2': (0, 1),
        }
        first_filter_size = int(100 / 2.0)  
        first_filter_stride = int(100 / 16.0) 
        self.cnn = nn.Sequential(
            nn.ConstantPad1d(self.padding_edf['conv1'], 0),  # conv1
            nn.Sequential(OrderedDict([
                ('conv1', nn.Conv1d(in_channels=1, out_channels=128, kernel_size=first_filter_size, stride=first_filter_stride,
                      bias=False))
            ])),
            nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.ConstantPad1d(self.padding_edf['max_pool1'], 0),  # max p 1
            nn.MaxPool1d(kernel_size=8, stride=8),
            nn.Dropout(p=0.5),
            nn.ConstantPad1d(self.padding_edf['conv2'], 0),  # conv2
            nn.Sequential(OrderedDict([
                ('conv2',
                 nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, bias=False))
            ])),
            nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.ConstantPad1d(self.padding_edf['conv2'], 0),  # conv3
            nn.Sequential(OrderedDict([
                ('conv3',nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, bias=False))
            ])),
            nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.ConstantPad1d(self.padding_edf['conv2'], 0),  # conv4
            nn.Sequential(OrderedDict([
                ('conv4', nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, bias=False))
            ])),
            nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.ConstantPad1d(self.padding_edf['max_pool2'], 0),  # max p 2
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Flatten(),
            nn.Dropout(p=0.5),
        )
        # 上面这段CNN的代码，是完全使用了参考代码的网络结构
        if network == "LSTM":
            self.rnn = nn.LSTM(input_size=2048, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=is_bidirectional)
        elif network == "GRU":
            self.rnn = nn.GRU(input_size=2048, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=is_bidirectional)
        elif network == "Attention":
        # encoder_layer = nn.TransformerEncoderLayer(d_model=2048, nhead=8, batch_first=True)
         # 这里其实用transformer_encoder也可以，但是如果叠加太多层，会跑不起来，而且发现使用attention的效果很一般
            self.qtrans = nn.Sequential(nn.Linear(2048,256), nn.ReLU(inplace=True))
            self.ktrans = nn.Sequential(nn.Linear(2048,256), nn.ReLU(inplace=True))
            self.vtrans = nn.Sequential(nn.Linear(2048,256), nn.ReLU(inplace=True))
            self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=4,  batch_first=True)

        self.rnn_dropout = nn.Dropout(p=0.5)
        if network != "Attention":
            # 使用双向RNN的时候，会有前向和后向的两个输出进行拼接作为最后的输出，因此大小需要×2
            self.fc = nn.Linear(hidden_size*2, 5) if is_bidirectional else nn.Linear(hidden_size, 5)
        else:
            self.fc = nn.Linear(256, 5)

        self.is_bidirectional = is_bidirectional
        self.hidden_size = hidden_size
        self.network = network
        self.seq_len = seq_len

    def forward(self, x):
        x = x.reshape(-1, 1, 3000)
        x = self.cnn(x)
        x = x.reshape(-1, self.seq_len, 2048)  # batch first == True
        assert x.shape[-1] == 2048
        if self.network != "Attention":
            x, _ = self.rnn(x)
            x = x.reshape(-1, self.hidden_size*2) if self.is_bidirectional else x.reshape(-1, self.hidden_size)
        else:
            # attention除了输出x外，还输出了attn_weights，也就是query和key运算后经类softmax的score，多头的话会进行平均
            x, _ = self.attention(self.qtrans(x),self.ktrans(x),self.vtrans(x))
            x = x.reshape(-1, 256)
        x = self.rnn_dropout(x)
        x = self.fc(x)

        return x

class ManualFeatureNet(nn.Module):
    """
    35-dimensional manual EEG features için sequence model.

    Input:
        x shape: [batch_size, seq_len, 35]

    Output:
        out shape: [batch_size * seq_len, 5]
    """

    def __init__(
        self,
        input_dim=35,
        hidden_dim=128,
        num_layers=1,
        num_classes=5,
        seq_len=20,
        is_bidirectional=True,
        dropout=0.3
    ):
        super(ManualFeatureNet, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.is_bidirectional = is_bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=is_bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )

        lstm_output_dim = hidden_dim * 2 if is_bidirectional else hidden_dim

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_output_dim, num_classes)

    def forward(self, x):
        """
        x shape:
            [batch_size, seq_len, 35]
        """

        batch_size = x.shape[0]

        # lstm_out shape:
        # [batch_size, seq_len, hidden_dim * direction]
        lstm_out, _ = self.lstm(x)

        lstm_out = self.dropout(lstm_out)

        # flatten sequence dimension
        # [batch_size, seq_len, hidden] -> [batch_size * seq_len, hidden]
        lstm_out = lstm_out.reshape(batch_size * self.seq_len, -1)

        # output shape:
        # [batch_size * seq_len, 5]
        out = self.classifier(lstm_out)

        return out

class FusionSleepNet(nn.Module):
    """
    Raw EEG + 35-dimensional manual feature fusion model.

    Inputs:
        x_raw shape    : [batch_size, seq_len, 3000]
        x_manual shape : [batch_size, seq_len, 35]

    Output:
        out shape      : [batch_size * seq_len, 5]
    """

    def __init__(
        self,
        seq_len=20,
        manual_dim=35,
        manual_proj_dim=128,
        lstm_hidden_dim=128,
        num_classes=5,
        is_bidirectional=True,
        dropout=0.3
    ):
        super(FusionSleepNet, self).__init__()

        self.seq_len = seq_len
        self.manual_dim = manual_dim
        self.manual_proj_dim = manual_proj_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_classes = num_classes
        self.is_bidirectional = is_bidirectional

        # Raw EEG CNN encoder
        # Input to CNN:
        # [batch_size * seq_len, 1, 3000]
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=50, stride=6),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=8),

            nn.Conv1d(64, 128, kernel_size=8, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 128, kernel_size=8, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        # CNN output after squeeze:
        # [batch_size * seq_len, 128]
        self.cnn_feature_dim = 128

        # Manual feature projection
        # [batch_size, seq_len, 35] -> [batch_size, seq_len, manual_proj_dim]
        self.manual_projection = nn.Sequential(
            nn.Linear(manual_dim, manual_proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Fused feature dimension
        fusion_dim = self.cnn_feature_dim + manual_proj_dim

        self.lstm = nn.LSTM(
            input_size=fusion_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=is_bidirectional
        )

        lstm_output_dim = lstm_hidden_dim * 2 if is_bidirectional else lstm_hidden_dim

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_output_dim, num_classes)

    def forward(self, x_raw, x_manual):
        """
        x_raw shape:
            [batch_size, seq_len, 3000]

        x_manual shape:
            [batch_size, seq_len, 35]
        """

        batch_size = x_raw.shape[0]

        # Raw EEG CNN branch
        # [batch_size, seq_len, 3000]
        # -> [batch_size * seq_len, 1, 3000]
        x_raw = x_raw.reshape(batch_size * self.seq_len, 1, 3000)

        cnn_feat = self.cnn(x_raw)

        # [batch_size * seq_len, 128, 1]
        # -> [batch_size * seq_len, 128]
        cnn_feat = cnn_feat.squeeze(-1)

        # [batch_size * seq_len, 128]
        # -> [batch_size, seq_len, 128]
        cnn_feat = cnn_feat.reshape(batch_size, self.seq_len, self.cnn_feature_dim)

        # Manual feature branch
        # [batch_size, seq_len, 35]
        # -> [batch_size, seq_len, manual_proj_dim]
        manual_feat = self.manual_projection(x_manual)

        # Fusion
        # [batch_size, seq_len, 128 + manual_proj_dim]
        fused = torch.cat([cnn_feat, manual_feat], dim=-1)

        # BiLSTM
        # [batch_size, seq_len, hidden_dim * direction]
        lstm_out, _ = self.lstm(fused)

        lstm_out = self.dropout(lstm_out)

        # [batch_size, seq_len, hidden]
        # -> [batch_size * seq_len, hidden]
        lstm_out = lstm_out.reshape(batch_size * self.seq_len, -1)

        # [batch_size * seq_len, 5]
        out = self.classifier(lstm_out)

        return out

class FusionAttentionSleepNet(nn.Module):
    """
    Raw EEG + manual feature fusion model with temporal attention.

    Inputs:
        x_raw shape    : [batch_size, seq_len, 3000]
        x_manual shape : [batch_size, seq_len, 35]

    Output:
        out shape      : [batch_size * seq_len, 5]

    Optional:
        attention_weights shape: [batch_size, seq_len]
    """

    def __init__(
        self,
        seq_len=20,
        manual_dim=35,
        manual_proj_dim=128,
        lstm_hidden_dim=128,
        num_classes=5,
        is_bidirectional=True,
        dropout=0.3
    ):
        super(FusionAttentionSleepNet, self).__init__()

        self.seq_len = seq_len
        self.manual_dim = manual_dim
        self.manual_proj_dim = manual_proj_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_classes = num_classes
        self.is_bidirectional = is_bidirectional

        # Raw EEG CNN encoder
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=50, stride=6),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=8),

            nn.Conv1d(64, 128, kernel_size=8, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 128, kernel_size=8, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.cnn_feature_dim = 128

        # Manual feature projection
        self.manual_projection = nn.Sequential(
            nn.Linear(manual_dim, manual_proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        fusion_dim = self.cnn_feature_dim + manual_proj_dim

        self.lstm = nn.LSTM(
            input_size=fusion_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=is_bidirectional
        )

        lstm_output_dim = lstm_hidden_dim * 2 if is_bidirectional else lstm_hidden_dim

        self.attention_layer = nn.Linear(lstm_output_dim, 1)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_output_dim * 2, num_classes)

    def forward(self, x_raw, x_manual, return_attention=False):
        """
        x_raw:
            [batch_size, seq_len, 3000]

        x_manual:
            [batch_size, seq_len, 35]
        """

        batch_size = x_raw.shape[0]

        # Raw EEG CNN branch
        x_raw = x_raw.reshape(batch_size * self.seq_len, 1, 3000)

        cnn_feat = self.cnn(x_raw)
        cnn_feat = cnn_feat.squeeze(-1)
        cnn_feat = cnn_feat.reshape(batch_size, self.seq_len, self.cnn_feature_dim)

        # Manual feature branch
        manual_feat = self.manual_projection(x_manual)

        # Fusion
        fused = torch.cat([cnn_feat, manual_feat], dim=-1)

        # BiLSTM
        lstm_out, _ = self.lstm(fused)
        lstm_out = self.dropout(lstm_out)

        # Attention scores
        # [batch_size, seq_len, hidden] -> [batch_size, seq_len, 1]
        attention_scores = self.attention_layer(lstm_out)

        # [batch_size, seq_len, 1]
        attention_weights = torch.softmax(attention_scores, dim=1)

        # Context vector:
        # [batch_size, hidden]
        context = torch.sum(attention_weights * lstm_out, dim=1)

        # Context'i her epoch tahmini için tekrar ediyoruz.
        # Böylece her zaman adımı hem kendi BiLSTM çıktısını hem de sequence context'ini kullanır.
        # [batch_size, hidden] -> [batch_size, seq_len, hidden]
        context_repeated = context.unsqueeze(1).repeat(1, self.seq_len, 1)

        # [batch_size, seq_len, hidden * 2]
        combined = torch.cat([lstm_out, context_repeated], dim=-1)

        # [batch_size, seq_len, hidden * 2]
        # -> [batch_size * seq_len, hidden * 2]
        combined = combined.reshape(batch_size * self.seq_len, -1)

        # [batch_size * seq_len, 5]
        out = self.classifier(combined)

        if return_attention:
            # [batch_size, seq_len]
            return out, attention_weights.squeeze(-1)

        return out

if __name__ == '__main__':
    from torchsummaryX import summary


    # 举例：运行模型（CNN+GRU）
    model_full = SleepNet(network="GRU", seq_len=32)
    summary(model_full, torch.randn(size=(20 * 32, 1, 3000)))  # batch_size=20, seq_len=32


