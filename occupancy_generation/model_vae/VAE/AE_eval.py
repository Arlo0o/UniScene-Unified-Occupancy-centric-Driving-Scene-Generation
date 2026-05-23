from turtle import forward
import torch
import torch.nn as nn
from copy import deepcopy

class Autoencoder_2D(nn.Module):
    def __init__(self,
        num_classes=18,
        expansion=4):
        super(Autoencoder_2D, self).__init__()
        # 编码器部分
        self.expansion = expansion
        self.num_cls = num_classes

        self.class_embeds = nn.Embedding(num_classes, expansion)
        self.encoder = nn.Sequential(
            nn.Conv2d(16*expansion, 32, kernel_size=3, stride=2, padding=1),  # 16x200x200 -> 32x100x100
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 32x100x100 -> 64x50x50
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 64x50x50 -> 128x25x25
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 128x25x25 -> 256x13x13
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 256x13x13 -> 512x7x7
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),  # 512x7x7 -> 512x4x4
            # nn.ReLU(),
            # nn.Conv2d(1024, 2048, kernel_size=3, stride=2, padding=1),  # 512x4x4 ->512x2x2

            nn.AvgPool2d(kernel_size=2)  # 
        )
        
        
        # self.fc_encoder = nn.Linear(1024*2*2, 2048)  # 将特征图展平并压缩到2048个特征

        # self.fc_decoder = nn.Linear(2048, 1024*4*4)
        # 解码器部分
        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1),  
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=0),  
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=0),  
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16*expansion, kernel_size=3, stride=2, padding=1, output_padding=1),  
            # nn.Sigmoid()  # 激活函数，确保输出在[0, 1]范围内
        )
    
    def forward_encoder(self, x):
        bs, F, H, W, D = x.shape
        x = self.class_embeds(x) # bs, F, H, W, D, c
        x = x.reshape(bs*F, H, W, D * self.expansion).permute(0, 3, 1, 2)
        x = self.encoder(x)

        return x

    def forward_decoder(self, z, x_shape):
        
        bs, F, H, W, D = x_shape
        x = z.reshape(z.size(0), 512, 2, 2)  # 重塑为适合解码器的形状
        x = self.decoder(x)
        # print(x.shape)
        logits = x.permute(0, 2, 3, 1).reshape(-1, D, self.expansion)
        template = self.class_embeds.weight.T.unsqueeze(0) # 1, expansion, cls
        similarity = torch.matmul(logits, template) # -1, D, cls
        return similarity.reshape(bs, F, H, W, D, self.num_cls)

    def forward_eval(self,x):
        z = self.forward_encoder(x)
        return z.reshape(z.size(0), -1)  

    def forward(self, x, metas):
        x_shape = x.shape
        z = self.forward_encoder(x)
        # print(z.shape)

        z = z.reshape(z.size(0), -1)  
        # print(z.shape)
        # x = self.fc_encoder(x)
        # mid z
        # x = self.fc_decoder(x)

        logits = self.forward_decoder(z, x_shape)
        

        output_dict = {}
        output_dict.update({'logits': logits})
        

        if not self.training:
            pred = logits.argmax(dim=-1).detach().cuda()
            output_dict['sem_pred'] = pred
            pred_iou = deepcopy(pred)
            
            pred_iou[pred_iou!=17] = 1
            pred_iou[pred_iou==17] = 0
            output_dict['iou_pred'] = pred_iou
            
        return output_dict



if __name__ == "__main__":
    # 创建模型实例
    model = Autoencoder_2D()

    input_tensor =  torch.randint(low=0, high=18, size=(2,10, 200, 200,16))

    output = model(input_tensor,0)

    print(output['logits'].shape)
