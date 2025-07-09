from .segface import SegFaceLapa_infer
from .cross_fusion import *
from .mobilefacenet import MobileFaceNet
from .ir50 import Backbone
from .spade_res import *
from utils import load_pretrained_weights

class SE_block(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()
        self.linear2 = torch.nn.Linear(input_dim, input_dim)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        x1 = self.linear1(x)
        x1 = self.relu(x1)
        x1 = self.linear2(x1)
        x1 = self.sigmod(x1)
        x = x * x1
        return x

class ClassificationHead(nn.Module):
    def __init__(self, input_dim: int, target_dim: int):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, target_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        y_hat = self.linear(x)
        return y_hat

class SegDownConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(11, 64, kernel_size=3, stride=4, padding=1),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
    
    def forward(self, x):
        x1 = self.layer1(x)  # [B, 64, 56, 56]
        x2 = self.layer2(x1)  # [B, 128, 28, 28]
        x3 = self.layer3(x2)  # [B, 256, 14, 14]
        x4 = self.layer4(x3)  # [B, 512, 7, 7]
        return x4


class ORSA(nn.Module):
    def __init__(self, img_size=224, num_classes=7, type="large"):
        super().__init__()
        depth = 8
        if type == "small":
            depth = 4
        if type == "base":
            depth = 6
        if type == "large":
            depth = 8

        self.img_size = img_size
        self.num_classes = num_classes

        # image
        self.ir_back = Backbone(50, 0.0, 'ir')
        ir_checkpoint = torch.load('models/pretrain/ir50.pth', map_location=lambda storage, loc: storage)
        self.ir_back = load_pretrained_weights(self.ir_back, ir_checkpoint)

        self.ir_layer = nn.Linear(1024,512)

        # landmark
        self.face_landback = MobileFaceNet([112, 112],136)
        face_landback_checkpoint = torch.load('models/pretrain/mobilefacenet_model_best.pth.tar', map_location=lambda storage, loc: storage)
        self.face_landback.load_state_dict(face_landback_checkpoint['state_dict'])
        
        for param in self.face_landback.parameters():
            param.requires_grad = False

        # segmentation
        self.face_segment = SegFaceLapa_infer(224, 'swin_base')
        face_segment_checkpoint = torch.load('models/pretrain/model_299.pt')
        self.face_segment.load_state_dict(face_segment_checkpoint['state_dict_backbone'])

        for param in self.face_segment.parameters():
            param.requires_grad = False

        self.seg_sample = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.seg_fusion = SPADEResnetBlock(512, 512, 512, norm_G='spectralinstance')

        # cross-fusion
        self.pyramid_fuse = HyVisionTransformer(in_chans=49, q_chanel = 49, embed_dim=512,
                                             depth=depth, num_heads=8, mlp_ratio=2.,
                                             drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1)


        self.se_block = SE_block(input_dim=512)
        self.head = ClassificationHead(input_dim=512, target_dim=self.num_classes)


    def forward(self, x):
        B_ = x.shape[0]
        x_face = F.interpolate(x, size=112)
        _, x_face = self.face_landback(x_face)
        x_face = x_face.view(B_, -1, 49).transpose(1,2)

        x_seg = self.face_segment(x)
        x_seg = self.seg_sample(x_seg)
        
        x_ir = self.ir_back(x).view(-1, 49, 1024)  
        x_ir = self.ir_layer(x_ir) 
        x_ir = x_ir.permute(0, 2, 1).view(B_, 512, 7, 7)

        x_ir = self.seg_fusion(x_ir, x_seg)
        x_ir = x_ir.view(B_, -1, 49).transpose(1,2)
        y_hat = self.pyramid_fuse(x_ir, x_face)

        y_hat = self.se_block(y_hat)
        y_feat = y_hat
        out = self.head(y_hat)

        return out, y_feat
    