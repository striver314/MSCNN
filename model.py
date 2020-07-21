import torch.nn as nn
import torch
from fullconnect import Batch_Net
import torch.nn.functional as F



class MSCNN(nn.Module):

    def __init__(self, windows, steps, skip, dims, C_nums, C_steps, cuda):

        super(MSCNN, self).__init__()
        self.CUDA= cuda
        self.skip = skip
        self.windows = windows
        self.dims = dims
        self.steps = steps
        self.days = int(self.windows/self.steps)
        self.C_nums = C_nums
        self.C_steps = C_steps
        self.pad = self.skip-(self.windows % (self.skip + 1))+1





        # CNN & CNN-skip
        self.shortcut = nn.Sequential(
            nn.Conv2d(self.windows, self.windows - self.C_steps + 1, kernel_size=(1, self.dims), stride=1, bias=False),
            nn.BatchNorm2d(self.windows - self.C_steps + 1)
        )
        self.CNN_normal = nn.Conv2d(1, self.C_nums, (self.C_steps, self.dims), stride=1)
        self.CNN_relu = nn.ReLU(inplace=True)
        self.CNN_expand_1 = nn.Conv2d(1, self.C_nums, (self.dims, 2), stride=(self.dims, 1))
        self.CNN_expand_2 = nn.Conv2d(1, self.C_nums, (self.dims, 3), stride=(self.dims, 1))
        self.CNN_expand_3 = nn.Conv2d(1, self.C_nums, (self.dims, 5), stride=(self.dims, 1))
        self.CNN_pool = nn.AdaptiveAvgPool2d((1, 1))

        # FM-SENet
        self.SENet_globavgpool = nn.AvgPool2d((1, 4), stride=1)
        self.fc1 = nn.Linear(in_features=self.C_nums, out_features=round(self.C_nums/10))
        self.fc2 = nn.Linear(in_features=round(self.C_nums/10), out_features=self.C_nums)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

        self.glo_aver_pool = torch.nn.AdaptiveAvgPool2d((self.C_nums, 1))
        self.fullconnect = Batch_Net(in_dim=self.C_nums*4,
                                     out_dim=self.dims,
                                     n_hidden_1=self.dims,
                                     n_hidden_2=self.dims)



    def forward(self, x):
        #Processing input Tensor
        batch_size = x.size(0)
        c1 = x.view(-1, 1, self.dims, self.windows).permute(0, 1, 3, 2)
        if self.pad > 0:
            if self.CUDA:
                pad = torch.zeros(batch_size, self.dims, self.pad).cuda()
            else:
                pad = torch.zeros(batch_size, self.dims, self.pad)
            c1_padding = torch.cat([c1.reshape(batch_size, self.dims, self.windows), pad], 2)
        else:
            c1_padding = c1

        #CNN
        c1_res = c1.permute(0, 2, 1, 3)
        c1_out = self.CNN_normal(c1)
        c1_out = F.relu(c1_out)
        c1_out = c1_out.permute(0, 2, 1, 3)
        c1_res = self.shortcut(c1_res)
        c1_out += c1_res
        c1_out.permute(0, 2, 1, 3)
        c1_out = c1_out.reshape(batch_size, self.C_nums, -1, 1)
        c1_out = self.CNN_pool(c1_out)

        # CNN_skip
        c2 = c1_padding.reshape(batch_size, 1, int(c1_padding.size(-1) / (self.skip + 1)),
                                self.dims * (self.skip + 1))
        c2 = c2.permute(0, 1, 3, 2)
        c2_out_1 = self.CNN_expand_1(c2)
        c2_out_2 = self.CNN_expand_2(c2)
        c2_out_3 = self.CNN_expand_3(c2)
        c2_out_1 = c2_out_1.permute(0, 1, 3, 2)
        c2_out_2 = c2_out_2.permute(0, 1, 3, 2)
        c2_out_3 = c2_out_3.permute(0, 1, 3, 2)
        c2_out_1 = F.relu(c2_out_1)
        c2_out_2 = F.relu(c2_out_2)
        c2_out_3 = F.relu(c2_out_3)



        #concat
        c2_out_1 = self.CNN_pool(c2_out_1)
        c2_out_2 = self.CNN_pool(c2_out_2)
        c2_out_3 = self.CNN_pool(c2_out_3)
        c2_out = torch.stack([c2_out_1, c2_out_2, c2_out_3]).squeeze(4)
        c2_out = c2_out.permute(1, 2, 0, 3)
        c_out = torch.cat([c1_out, c2_out], 2).squeeze(3)

        #SENet
        original_out =c_out
        c_out = self.SENet_globavgpool(c_out)
        c_out = c_out.permute(0, 2, 1)
        c_out = self.fc1(c_out)
        c_out = self.relu(c_out)
        c_out = self.fc2(c_out)
        c_out = self.sigmoid(c_out)
        c_out = c_out.permute(0, 2, 1)
        c_out = c_out * original_out

        #Dense
        c = c_out.reshape(batch_size, -1)
        c = torch.squeeze(c)
        c = self.fullconnect(c)
        c = torch.squeeze(c)
        return c