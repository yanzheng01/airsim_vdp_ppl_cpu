import torch
import airsim
import numpy as np
import io
from PIL import Image

from model import Decoder, rgb_stem
from cfg import DEVICE

import matplotlib.pyplot as plt




# decoder = Decoder(z_dim=50, rgb_stem=rgb_stem, hidden_dim=400)
# # z = torch.zeros(batch_size, 50, device=DEVICE)
# # print(decoder)

# decoder.load_state_dict(torch.load('decoder_weights.pt', map_location='cpu'))
# decoder.to(DEVICE)

class CarmeraRgbDepthRender:
    def __init__(self, client, camera_name, z_dim=50, rgb_stem=rgb_stem, hedden_dim=400, decoder_weights_path='decoder_weights.pt'):
        self.client = client
        self.camera_name = camera_name
        self.z_dim = z_dim
        self.z = torch.zeros(1, self.z_dim, device=DEVICE)
        self.rgb_stem = rgb_stem
        self.hidden_dim = hedden_dim
        self.decoder_weights_path = decoder_weights_path
        self.init_decoder()

    def init_decoder(self):
        self.decoder = Decoder(z_dim=self.z_dim, rgb_stem=self.rgb_stem, hidden_dim=self.hidden_dim)
        self.decoder.load_state_dict(torch.load(self.decoder_weights_path, map_location='cpu'))
        self.decoder.to(DEVICE)
    
    def infer(self, rgb):
        self.decoder.eval()
        with torch.no_grad():
            img_param = self.decoder(self.z, rgb)
        return img_param
    
    def render(self):
        rgb_response = self.client.simGetImage(
                        camera_name=self.camera_name,
                        image_type=airsim.ImageType.Scene
                    )
        rgb_img = airsim.string_to_uint8_array(rgb_response)
        rgb_img = Image.open(io.BytesIO(rgb_img.tobytes()))
        rgb_img = rgb_img.resize((64, 48))
        rgb_img = np.array(rgb_img)
        if rgb_img.shape[2] == 4:
            rgb_img = rgb_img[:, :, :3]
        elif rgb_img.shape[2] == 1:
            rgb_img = np.repeat(rgb_img, 3, axis=2)
        x = torch.tensor(rgb_img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        img = self.infer(x.unsqueeze(0).to(DEVICE))
        return img
        
    
# def infer(rgb):
#     decoder.eval()
#     with torch.no_grad():
#         batch_size = rgb.size(0)
        
#         img_param = decoder(z, rgb)
#     return img_param


# def render_depth(client, camera_name):
#     rgb_response = client.simGetImage(
#                     camera_name=camera_name,
#                     image_type=airsim.ImageType.Scene
#                 )
#     #print(rgb_response)
#     rgb_img = airsim.string_to_uint8_array(rgb_response)
#     rgb_img = Image.open(io.BytesIO(rgb_img.tobytes()))
#     rgb_img = rgb_img.resize((64, 48))
#     rgb_img = np.array(rgb_img)
#     if rgb_img.shape[2] == 4:
#         rgb_img = rgb_img[:, :, :3]
#     elif rgb_img.shape[2] == 1:
#         rgb_img = np.repeat(rgb_img, 3, axis=2)
#     x = torch.tensor(rgb_img, dtype=torch.float32).permute(2, 0, 1) / 255.0
#     # x = x.unsqueeze(0)  # Add batch dimension
#     img = infer(x.unsqueeze(0).to(DEVICE))
#     # depth_img2d = img.view(12*sr, 16*sr).cpu

#     return img

if __name__ == "__main__":
    # import io
    client = airsim.MultirotorClient(ip='192.168.1.2', port=41452) # 连接AirSim仿真环境
    client.confirmConnection()
    client.reset()
    carmer = CarmeraRgbDepthRender(client, camera_name="0")

    depth_img = carmer.render()
    print(depth_img)

    fig = plt.figure()
    plt.imshow(depth_img.view(12,16).cpu(), cmap='gray')
    plt.show()
