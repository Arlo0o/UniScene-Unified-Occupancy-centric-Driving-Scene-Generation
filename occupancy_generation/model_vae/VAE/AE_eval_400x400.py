from turtle import forward
import torch
import torch.nn as nn
from copy import deepcopy

class Autoencoder_2D_400x400(nn.Module):
    def __init__(self,
        num_classes=18,
        expansion=4):
        super(Autoencoder_2D_400x400, self).__init__()
        # 编码器部分 - 适配 400x400x32 输入
        self.expansion = expansion
        self.num_cls = num_classes

        self.class_embeds = nn.Embedding(num_classes, expansion)
        self.encoder = nn.Sequential(
            # 32*expansion -> 64: 400x400 -> 200x200
            nn.Conv2d(32*expansion, 64, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),
            # 64 -> 128: 200x200 -> 100x100
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),
            # 128 -> 256: 100x100 -> 50x50
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),
            # 256 -> 512: 50x50 -> 25x25
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),
            # 512 -> 512: 25x25 -> 13x13
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),
            # 512 -> 1024: 13x13 -> 7x7
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),
            # 1024 -> 1024: 7x7 -> 4x4 (停在这里，保持4x4)
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),  
        )
        
        # 解码器部分 - 从 1024x4x4 还原到 400x400x32
        self.decoder = nn.Sequential(
            # 1024 -> 1024: 4x4 -> 7x7
            nn.ConvTranspose2d(1024, 1024, kernel_size=3, stride=2, padding=1, output_padding=0),  
            nn.ReLU(),
            # 1024 -> 512: 7x7 -> 13x13
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=0),  
            nn.ReLU(),
            # 512 -> 512: 13x13 -> 25x25
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.ReLU(),
            # 512 -> 256: 25x25 -> 50x50
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  
            nn.ReLU(),
            # 256 -> 128: 50x50 -> 100x100
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  
            nn.ReLU(),
            # 128 -> 64: 100x100 -> 200x200
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  
            nn.ReLU(),
            # 64 -> 32*expansion: 200x200 -> 400x400
            nn.ConvTranspose2d(64, 32*expansion, kernel_size=3, stride=2, padding=1, output_padding=1),  
        )
    
    def forward_encoder(self, x):
        bs, F, H, W, D = x.shape  # 期望: (bs, F, 400, 400, 32)
        x = self.class_embeds(x) # bs, F, H, W, D, expansion
        x = x.reshape(bs*F, H, W, D * self.expansion).permute(0, 3, 1, 2)  # (bs*F, 32*expansion, 400, 400)
        x = self.encoder(x)  # (bs*F, 1024, 4, 4)
        return x

    def forward_decoder(self, z, x_shape):
        bs, F, H, W, D = x_shape  # (bs, F, 400, 400, 32)
        x = z.reshape(z.size(0), 1024, 4, 4)  # 重塑为适合解码器的形状 1024x4x4
        x = self.decoder(x)  # (bs*F, 32*expansion, 400, 400)
        
        # 转换回分类logits
        # x: (bs*F, 32*expansion, 400, 400) -> (bs*F, 400, 400, 32*expansion)
        logits = x.permute(0, 2, 3, 1).reshape(-1, D, self.expansion)  # (-1, 32, expansion)
        template = self.class_embeds.weight.T.unsqueeze(0) # (1, expansion, num_classes)
        similarity = torch.matmul(logits, template) # (-1, 32, num_classes)
        return similarity.reshape(bs, F, H, W, D, self.num_cls)  # (bs, F, 400, 400, 32, num_classes)

    def forward_eval(self, x):
        z = self.forward_encoder(x)
        return z.reshape(z.size(0), -1)  

    def forward(self, x, metas):
        x_shape = x.shape
        z = self.forward_encoder(x)
        
        # 展平潜在表示
        z = z.reshape(z.size(0), -1)  
        
        # 解码
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
    model = Autoencoder_2D_400x400()
    
    # 测试 400x400x32 输入
    input_tensor = torch.randint(low=0, high=18, size=(2, 10, 400, 400, 32))
    
    print(f"Input shape: {input_tensor.shape}")
    
    # 先测试编码器
    z_enc = model.forward_encoder(input_tensor)
    print(f"Encoder output shape: {z_enc.shape}")
    
    # 测试完整前向传播
    try:
        output = model(input_tensor, 0)
        print(f"Output logits shape: {output['logits'].shape}")
        print("Expected output shape: (2, 10, 400, 400, 32, 18)")
    except Exception as e:
        print(f"Error in forward pass: {e}")
    
    # 测试编码器单独使用
    z = model.forward_eval(input_tensor)
    print(f"Encoded feature shape: {z.shape}")
