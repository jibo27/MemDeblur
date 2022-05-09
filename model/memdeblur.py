import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from thop import profile

from .modules import *

class Model(nn.Module):
    def __init__(self, para):
        super().__init__()

        self.mid_channels = para.mid_channels
        self.mem_every = para.mem_every
        self.num_blocks_forward = para.num_blocks_forward
        self.num_blocks_backward = para.num_blocks_backward
        self.scales = para.scales


        # ----------------- Deblurring branch ----------------- 
        # Downsample Module
        self.n_feats = 16
        self.downsampling = nn.Sequential(
            conv5x5(3, 3, stride=1),
            RDB_DS(in_channels=3, growthRate=16, num_layer=3),
            RDB_DS(in_channels=4 * 3, growthRate=int(self.n_feats * 3 / 2), num_layer=3)
        )
        self.upsampling = nn.Sequential(
            nn.ConvTranspose2d(self.mid_channels, 2 * self.n_feats, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.ConvTranspose2d(2 * self.n_feats, self.n_feats, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            conv5x5(self.n_feats, 3, stride=1)
        )

        # Feature Extraction Module
        self.forward_input_convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(self.mid_channels + 48 * 2 + 64 * 3, self.mid_channels, 3, 1, 1, bias=True),
                          nn.LeakyReLU(negative_slope=0.1, inplace=True)),
            nn.Sequential(nn.Conv2d(self.mid_channels + 48 * 3 + 64 * 3, self.mid_channels, 3, 1, 1, bias=True),
                          nn.LeakyReLU(negative_slope=0.1, inplace=True)),
            nn.Sequential(nn.Conv2d(self.mid_channels + 48 * 3 + 64 * 3, self.mid_channels, 3, 1, 1, bias=True),
                          nn.LeakyReLU(negative_slope=0.1, inplace=True)),
        ])
        self.forward_resblocks = ResidualBlocksWithoutInputConv(
            self.mid_channels, self.num_blocks_forward) # 32 for memory network

        self.backward_resblocks = ResidualBlocksWithInputConv(
            self.mid_channels * 2 + 48 * 2 , self.mid_channels, self.num_blocks_backward) 

        # Upsample Module
        self.fusion = nn.Conv2d(
            self.mid_channels * 2, self.mid_channels, 1, 1, 0, bias=True)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)


        # ----------------- Memory branch ----------------- 
        self.memory = Memory(para)


    def forward(self, inputs, profile_flag=False):
        if profile_flag:
            return self.profile_forward(inputs)
        self.memory.mem_bank_forward.training = self.training
        self.memory.mem_bank_forward.clear_memory()
        self.memory.mem_bank_backward.training = self.training
        self.memory.mem_bank_backward.clear_memory()

        n, t, c, h_ori, w_ori = inputs.size()
        n, t = inputs.shape[:2]

        outputs = [[] for _ in range(len(self.scales))]

        downsampled_features = [] 
        keys = []
        for i in range(0, t):
            downsampled_features.append(self.downsampling(inputs[:,i]))
            keys.append(self.memory.encode_key(downsampled_features[-1]))

        # ------------ backward ------------ 
        feat_backs = []
        h = h_ori // 4
        w = w_ori // 4
        for i in range(t - 1, -1, -1):
            downsampled_feature = downsampled_features[i]
            key_curr = keys[i]

            if i == t - 1:
                feat = torch.cat([downsampled_feature, downsampled_feature.new_zeros(n, self.mid_channels*2+downsampled_feature.shape[1], h, w)], dim=1)
            else:
                ret_match = self.memory.mem_bank_backward.match_memory(key_curr)
                feat = ret_match[0]
                feat = self.memory.decoder_feat(feat)[...,:h,:w]
                feat = torch.cat([downsampled_feature, downsampled_features[i + 1], feat, prev_feat], dim=1)

            feat = self.backward_resblocks(feat)

            if self.training:
                if i in list(range(t - 1, -1, -self.mem_every[-1])) + [t - 1]:
                    self.memory.mem_bank_backward.add_memory(key_curr, self.memory.encode_value_feat(downsampled_feature, feat))
            else:
                if i in list(range(2, t, self.mem_every[-1])) + [t - 1]:
                    self.memory.mem_bank_backward.add_memory(key_curr, self.memory.encode_value_feat(downsampled_feature, feat))
            feat_backs.append(feat)
            prev_feat = feat
        feat_backs = list(reversed(feat_backs))


        # ----------- Forward pass ---------
        for s, scale in enumerate(self.scales):
            if scale != 1: # need to downsample
                h_curr = h_ori // scale
                w_curr = w_ori // scale
                h_curr = 4 * (h_curr // 4) + (4 if h_curr % 4 != 0 else 0)
                w_curr = 4 * (w_curr // 4) + (4 if w_curr % 4 != 0 else 0)
            else:
                h_curr = h_ori
                w_curr = w_ori
            h = h_curr // 4
            w = w_curr // 4
            for i in range(0, t):

                if s != 0:
                    inputs_du = F.interpolate(input=outputs[s-1][i], size=(h_curr, w_curr), mode='bilinear', align_corners=False)
                    downsampled_feature_du = self.downsampling(inputs_du)

                if scale != 1: # need to downsample
                    inputs_curr = F.interpolate(input=inputs[:, i], size=(h_curr, w_curr), mode='bilinear', align_corners=False)
                    downsampled_feature = self.downsampling(inputs_curr)
                    key_curr = self.memory.encode_key(downsampled_feature) # (B, T=3, C=3, H, W)

                    # Features from backward
                    scaled_feat_back = F.interpolate(feat_backs[i], size=(h, w), mode='bilinear', align_corners=False)
                else:
                    inputs_curr = inputs[:, i]
                    downsampled_feature = downsampled_features[i]
                    key_curr = keys[i]

                    # Features from backward
                    scaled_feat_back = feat_backs[i]

                # Memory from backward
                feat_future = self.memory.mem_bank_backward.match_memory(key_curr)[0]
                feat_future = self.memory.decoder_feat(feat_future)
                feat_future = feat_future[...,:h,:w]

                if i == 0:
                    if s == 0:
                        feat = torch.cat([downsampled_feature, feat_future, downsampled_feature.new_zeros(n, self.mid_channels * 3 + 48, h, w)], dim=1)
                    else:
                        feat = torch.cat([downsampled_feature, feat_future, downsampled_feature_du, downsampled_feature.new_zeros(n, self.mid_channels * 3 + 48, h, w)], dim=1)
                else:
                    ret_match = self.memory.mem_bank_forward.match_memory(key_curr)
                    mem, feat = ret_match[0], ret_match[1]
                    feat = self.memory.decoder_feat(feat)[...,:h,:w]
                    decoded_mem = self.memory.decoder_mem(mem)[...,:h,:w]
                    if s == 0:
                        feat = torch.cat([downsampled_feature, feat_future, prev_downsampled_feature, feat, prev_feat, decoded_mem], dim=1)
                    else:
                        feat = torch.cat([downsampled_feature, feat_future, downsampled_feature_du, prev_downsampled_feature, feat, prev_feat, decoded_mem], dim=1)

                feat = self.forward_input_convs[s](feat)
                feat = self.forward_resblocks(feat)

                out = torch.cat([scaled_feat_back, feat], dim=1)
                out = self.lrelu(self.fusion(out))
                out = self.upsampling(out)
                out += inputs_curr
                outputs[s].append(out)

                if ((i % self.mem_every[s]) == 0):
                    self.memory.mem_bank_forward.add_memory(key_curr, [self.memory.encode_value_mem(downsampled_feature, self.downsampling(out)), self.memory.encode_value_feat(downsampled_feature, feat)])
                prev_downsampled_feature = downsampled_feature
                prev_feat = feat

        results = [torch.stack(outputs[s], dim=1) for s in range(len(self.scales))]
        return results

    def profile_forward(self, inputs):
        return self.forward(inputs)


class ResidualBlocksWithInputConv(nn.Module):
    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()

        main = []

        # a convolution used to match the channels of the residual blocks
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # residual blocks
        main.append(
            make_layer(
                ResidualBlockNoBN, num_blocks, mid_channels=out_channels))

        self.main = nn.Sequential(*main)

    def forward(self, feat):
        return self.main(feat)


class ResidualBlocksWithoutInputConv(nn.Module):
    def __init__(self, out_channels=64, num_blocks=30):
        super().__init__()

        main = []

        # residual blocks
        main.append(
            make_layer(
                ResidualBlockNoBN, num_blocks, mid_channels=out_channels))

        self.main = nn.Sequential(*main)

    def forward(self, feat):
        return self.main(feat)




class KeyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.forward_resblocks = ResidualBlocksWithInputConv(48, 48, 2)
        resnet = resnet50(pretrained=True, extra_chan=45)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1

    def forward(self, f):
        x = self.forward_resblocks(f)
        x = self.conv1(x) 
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        f4 = self.layer1(x)

        return f4


class KeyProjection(nn.Module):
    def __init__(self, indim, keydim):
        super().__init__()
        self.key_proj = nn.Conv2d(indim, keydim, kernel_size=3, padding=1)

        nn.init.orthogonal_(self.key_proj.weight.data)
        nn.init.zeros_(self.key_proj.bias.data)
    
    def forward(self, x):
        return self.key_proj(x)

class ValueEncoder(nn.Module):
    def __init__(self, in_channels=96):
        super().__init__()
        self.forward_resblocks = ResidualBlocksWithInputConv(in_channels, 32, 2)
        resnet = resnet50(pretrained=True, extra_chan=32- 3)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1

    def forward(self, image, gt):
        f = torch.cat([image, gt], 1)

        x = self.forward_resblocks(f)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)

        return x


class Decoder(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, scale_factor=4):
        super().__init__()
        self.scale_factor = scale_factor # 4

        self.forward_resblocks = ResidualBlocksWithInputConv(in_channels, out_channels, 2) 

        self.upsample = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * (4**2), kernel_size=3, padding=1),
            nn.PixelShuffle(self.scale_factor),
        )

    def forward(self, x):
        x = self.forward_resblocks(x)
        return self.upsample(x) + F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)


class Memory(nn.Module):
    def __init__(self, para):
        super().__init__()
        self.key_encoder = KeyEncoder() 
        self.value_encoder_mem = ValueEncoder() 
        self.value_encoder_feat = ValueEncoder(112) 

        self.key_proj = KeyProjection(256, keydim=64)

        self.decoder_mem = Decoder(in_channels=256, out_channels=64)
        self.decoder_feat = Decoder(in_channels=256, out_channels=64)

        self.mem_bank_forward = MemoryBank(para, num_values=2)
        self.mem_bank_backward = MemoryBank(para, num_values=1)


    def encode_value_mem(self, frame, gts): 
        f16 = self.value_encoder_mem(frame, gts)

        return f16.unsqueeze(2)

    def encode_value_feat(self, frame, gts): 
        f16 = self.value_encoder_feat(frame, gts)

        return f16.unsqueeze(2)
 

    def encode_key(self, frame):
        f4 = self.key_encoder(frame)
        k4 = self.key_proj(f4)
        return k4


def softmax_w_top(x, top):
    top = min(top, x.shape[1])
    values, indices = torch.topk(x, k=top, dim=1)
    x_exp = torch.softmax(values, dim=1)
    x.zero_().scatter_(1, indices, x_exp)

    return x



class MemoryBank:                                                           
    def __init__(self, para, num_values, top_k=20):
        self.top_k = top_k                                                  

        self.CK = None
        self.CV = None
        self.num_values = num_values
        self.test_mem_length = para.test_mem_length

        self.mem_k = None                                                   
        self.mem_vs = [None for i in range(self.num_values)]

    def _global_matching(self, mk, qk):                                     
        B, CK, NE = mk.shape                                                
                                                                            
        a = mk.pow(2).sum(1).unsqueeze(2)                                   
        b = 2 * (mk.transpose(1, 2) @ qk)                                   

        affinity = (-a+b) / math.sqrt(CK)  # B, NE, HW; [B, [256|512|...], 256]
        if self.training:
            maxes = torch.max(affinity, dim=1, keepdim=True)[0]
            x_exp = torch.exp(affinity - maxes)
            x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
            affinity = x_exp / x_exp_sum 
        else:
            affinity = softmax_w_top(affinity, top=self.top_k)  # B, THW, HW    
                                                                            
        return affinity
                                                                            
    def _readout(self, affinity, mv):                                       
        return torch.bmm(mv, affinity)                                      
                                                                            
    def match_memory(self, qk):                                             
        b, c_k, h, w = qk.shape

        qk = qk.flatten(start_dim=2)

        mvs = []

        if self.temp_k is not None:
            mk = torch.cat([self.mem_k, self.temp_k], 2)
            for i in range(self.num_values):
                mvs.append(torch.cat([self.mem_vs[i], self.temp_vs[i]], 2))

        else:
            mk = self.mem_k 
            mvs = self.mem_vs

        affinity = self._global_matching(mk, qk)

        readout_mems = []
        for i in range(self.num_values):
            readout_mems.append(self._readout(affinity, mvs[i]))

        return [readout_mems[i].view(b, self.CV, h, w) for i in range(self.num_values)]

    def add_memory(self, key, values, is_temp=False):
        if not isinstance(values, list):
            values = [values]
        self.temp_k = None
        self.temp_vs = [None for _ in range(self.num_values)]
        key = key.flatten(start_dim=2)                                      
        values = [values[i].flatten(start_dim=2) for i in range(self.num_values)]
                                                                            
        if self.mem_k is None:                                              
            self.mem_k = key
            self.mem_vs = values
            self.CK = key.shape[1]
            self.CV = values[0].shape[1]
        else:                                                               
            if is_temp:                                                     
                self.temp_k = key
                self.temp_vs = values
            else:
                self.mem_k = torch.cat([self.mem_k, key], 2)
                self.mem_vs = [torch.cat([self.mem_vs[i], values[i]], 2) for i in range(self.num_values)]

                if (not self.training) and (self.test_mem_length is not None):
                    self.mem_k = self.mem_k[..., -key.shape[-1]*self.test_mem_length:]
                    self.mem_vs = [self.mem_vs[i][..., -values[0].shape[-1]*self.test_mem_length:]  for i in range(self.num_values)]


    def clear_memory(self):
        self.mem_k = None
        self.mem_vs = [None for i in range(self.num_values)]



                                                                            
def cost_profile(model, H, W, seq_length=100):
    x = torch.randn(1, seq_length, 3, H, W).cuda()
    profile_flag = True
    flops, params = profile(model, inputs=(x, profile_flag), verbose=False)

    return flops / seq_length, params

def feed(model, iter_samples):
    inputs = iter_samples[0]
    outputs = model(inputs)
    return outputs