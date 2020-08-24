import os

import torch
from PIL import Image, ImageDraw
from torchvision import transforms
import torchvision

from data_loader import get_loader
from network import U_Net, R2U_Net, AttU_Net, R2AttU_Net, NestedUNet, UNet
from iter_net.iternet_model import Iternet, AttUIternet, R2UIternet
from AG_Net.core.models import AG_Net

from torchvision import transforms as T


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

# Gets the predicted patches
test_loader = get_loader(image_path="test_patches/STARE/train/",
                            image_size=48,
                            batch_size=1,
                            num_workers=0,
                            mode='test',
                            augmentation_prob=0.)


for i, (images, GT, image_path) in enumerate(test_loader):
    model = Iternet(1, 1)
    model.load_state_dict(torch.load('./models/STARE/Iternet-50-0.0020-10-0.5230_Index_Combo_Dropout_STAREIndex.pkl',
                                     map_location=torch.device('cpu')))
    model.train(False)
    model.eval()

    SR = model(images)

    torchvision.utils.save_image(SR.data.cpu(), 'result/test_output/STARE/Iternet/%s' % image_path)


results = []
for i in range(1,15):
    DATA_RAW_DIR = "./data/STARE"
    IOSTAR_IMAGE_TEST = DATA_RAW_DIR + "/train_GT"

    image = Image.open(IOSTAR_IMAGE_TEST + "/" + str(i).zfill(5) + ".ppm")
    width, height = image.size

    rounded_width = 48 * (width // 48)
    rounded_height = 48 * (height // 48)

    trimmed_data = image.crop((0, 0, rounded_width, rounded_height))
    trimmed_image = Image.new('RGB', (rounded_width, rounded_height), 255)
    trimmed_image.paste(trimmed_data)
    slide_image = trimmed_image
    slide_width, slide_height = slide_image.size

    new_image = Image.new('RGB', slide_image.size, 0)
    new_true_GT = Image.new('RGB', slide_image.size, 0)

    # Split and save
    patch_size = 48
    for i_x in range(slide_width // patch_size):
        for i_y in range(slide_height // patch_size):
            # print(str(i_x).zfill(2))
            # print(str(i_y).zfill(2))

            patch_image = Image.open(
                "./result/test_output/STARE/Iternet/" + str(i).zfill(5) + "_x" + str(i_x).zfill(2) + "_y" + str(i_y).zfill(2) + ".png")
            true_GT = Image.open(
                "./test_patches/STARE/train_GT/" + str(i).zfill(5) + "_x" + str(i_x).zfill(2) + "_y" + str(i_y).zfill(2) + ".png")
            # black_image =
            x = patch_size * i_x
            y = patch_size * i_y
            box = (x, y, x + patch_size, y + patch_size)
            new_image.paste(patch_image, box)
            new_true_GT.paste(true_GT, box)

    new_image.save("result/test_whole_image/STARE/Iternet/" + str(i).zfill(5) + ".png")
    new_true_GT.save("result/test_whole_image_true/STARE/" + str(i).zfill(5) + ".png")

    from evaluation import *

    SR = Image.open("result/test_whole_image/STARE/Iternet/" + str(i).zfill(5) + ".png")
    GT = Image.open("result/test_whole_image_true/STARE/" + str(i).zfill(5) + ".png")

    Transform = []
    Transform.append(T.ToTensor())
    Transform = T.Compose(Transform)
    SR = Transform(SR)
    # print(torch.max(SR[3]))

    GT = Transform(GT)
    print(torch.max(GT[0]))

    acc = get_accuracy(SR[0], GT[0])
    sensitivity = get_sensitivity(SR[0], GT[0])
    specificity = get_specificity(SR[0], GT[0])
    dice = get_DC(SR[0], GT[0])
    jaccard = get_JS(SR[0], GT[0])
    results.append({'acc': acc,
                    'sensitivity': sensitivity,
                    'specificity': specificity,
                    'dice': dice,
                    'jaccard': jaccard})
    print(get_accuracy(SR[0], GT[0]))
    print(get_sensitivity(SR[0], GT[0]))
    print(get_specificity(SR[0], GT[0]))
    print(get_DC(SR[0], GT[0]))
    print(get_JS(SR[0], GT[0]))

print(results)

# DATA_RAW_DIR = "./data/CHASEDB1"
# IOSTAR_IMAGE_TEST = DATA_RAW_DIR + "/train"
#
# image = Image.open(IOSTAR_IMAGE_TEST + "/00001.jpg")
# width, height = image.size
#
# rounded_width = 48 * (width // 48)
# rounded_height = 48 * (height // 48)
#
# trimmed_data = image.crop((0, 0, rounded_width, rounded_height))
# trimmed_image = Image.new('RGB', (rounded_width, rounded_height), 255)
# trimmed_image.paste(trimmed_data)
# slide_image = trimmed_image
# slide_width, slide_height = slide_image.size
#
# new_image = Image.new('RGB', slide_image.size, 0)
# new_true_GT = Image.new('RGB', slide_image.size, 0)
#
#
# # Split and save
# patch_size = 48
# for i_x in range(slide_width//patch_size):
#     for i_y in range(slide_height//patch_size):
#         print(str(i_x).zfill(2))
#         print(str(i_y).zfill(2))
#
#         patch_image = Image.open("./result/test_output/CHASEDB1/AttUNet/00001_x" + str(i_x).zfill(2) +  "_y" + str(i_y).zfill(2) + ".png")
#         true_GT = Image.open("./test_patches/CHASEDB1/train_GT/00001_x" + str(i_x).zfill(2) +  "_y" + str(i_y).zfill(2) + ".png")
#         # black_image =
#         x = patch_size * i_x
#         y = patch_size * i_y
#         box = (x, y, x + patch_size, y + patch_size)
#         new_image.paste(patch_image, box)
#         new_true_GT.paste(true_GT, box)
#
# new_image.save("result/test_whole_image/CHASEDB1/AttUNet/00001.png")
# new_true_GT.save("result/test_whole_image_true/CHASEDB1/00001.png")
#
#
# from evaluation import *
#
# SR = Image.open("result/test_whole_image/CHASEDB1/AttUNet/00001.png")
# GT = Image.open("result/test_whole_image_true/CHASEDB1/00001.png")
#
# Transform = []
# Transform.append(T.ToTensor())
# Transform = T.Compose(Transform)
# SR = Transform(SR)
# # print(torch.max(SR[3]))
#
# GT = Transform(GT)
# print(torch.max(GT[0]))
#
# print(get_accuracy(SR[0], GT[0]))
# print(get_sensitivity(SR[0], GT[0]))
# print(get_specificity(SR[0], GT[0]))
# print(get_DC(SR[0], GT[0]))
# print(get_JS(SR[0], GT[0]))





