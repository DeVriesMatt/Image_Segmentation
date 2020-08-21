import os

import torch
from PIL import Image
from torchvision import transforms
import torchvision

from data_loader import get_loader
from network import U_Net, R2U_Net, AttU_Net, R2AttU_Net, NestedUNet, UNet
from iter_net.iternet_model import Iternet, AttUIternet, R2UIternet
from AG_Net.core.models import AG_Net


# class Tester(object):
#     def __init__(self, model_type, model_path, test_loader):
#         self.test_loader = test_loader
#         self.unet_path = model_path
#         self.model_type = model_type
#
#     def build_model(self):
#         """Build generator and discriminator."""
#         if self.model_type == 'UNet':
#             self.unet = UNet(n_channels=1, n_classes=1)
#         elif self.model_type == 'R2U_Net':
#             self.unet = R2U_Net(img_ch=1, output_ch=1, t=self.t)  # TODO: changed for green image channel
#         elif self.model_type == 'AttU_Net':
#             self.unet = AttU_Net(img_ch=1, output_ch=1)
#         elif self.model_type == 'R2AttU_Net':
#             self.unet = R2AttU_Net(img_ch=1, output_ch=1, t=self.t)
#         elif self.model_type == 'Iternet':
#             self.unet = Iternet(n_channels=1, n_classes=1)
#         elif self.model_type == 'AttUIternet':
#             self.unet = AttUIternet(n_channels=1, n_classes=1)
#         elif self.model_type == 'R2UIternet':
#             self.unet = R2UIternet(n_channels=1, n_classes=1)
#         elif self.model_type == 'NestedUNet':
#             self.unet = NestedUNet(in_ch=1, out_ch=1)
#         elif self.model_type == "AG_Net":
#             self.unet = AG_Net(n_classes=1, bn=True, BatchNorm=False)
#
#     def test_model(self, image, x, y):
#         self.build_model()
#         self.unet.load_state_dict(torch.load(self.unet_path))
#
#         self.unet.train(False)
#         self.unet.eval()
#
#         transform = transforms.Compose([
#             transforms.ToTensor()
#         ])
#
#         path = 'test_patches/DRIVE/test/{0}_x{1}_y{2}.png'.format(image, x, y)
#         image_arr = transform(Image.open(path))
#         image_tensor = torch.Tensor(image_arr).unsqueeze(0)
#         SR = self.unet(images)


test_loader = get_loader(image_path="test_patches/DRIVE/test/",
                            image_size=48,
                            batch_size=1,
                            num_workers=0,
                            mode='test',
                            augmentation_prob=0.)



for i, (images, GT, image_path) in enumerate(test_loader):
    model = UNet(n_channels=1, n_classes=1)
    model.load_state_dict(torch.load('./models/UNet-80-0.0020-15-0.4000.pth'))
    model.train(False)
    model.eval()

    SR = model(images)

    torchvision.utils.save_image(SR.data.cpu(),
                                 os.path.join('result/test_output',
                                              '{}.png'.format(image_path)))





