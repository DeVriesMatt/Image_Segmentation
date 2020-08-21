import os
import random
from random import shuffle
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image
import cv2


class ImageFolder(data.Dataset):
	def __init__(self, root, image_size=48, mode='train', augmentation_prob=0.4):  # TODO: change image size for patch
		"""Initializes image paths and preprocessing module."""
		self.root = root

		# GT : Ground Truth
		self.GT_paths = root[:-1] + '_GT/'  # TODO: change for non image patches self.GT_paths = root[:-1]+'_GT/'
		self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))  # root[:-1]+'_GT/'  # TODO: test on only GT images as input list(map(lambda x: os.path.join(root, x), os.listdir(root)))
		self.image_size = image_size
		self.mode = mode
		self.RotationDegree = [0, 90, 180, 270]
		self.augmentation_prob = augmentation_prob
		print("image count in {} path :{}".format(self.mode, len(self.image_paths)))

	def __getitem__(self, index):
		"""Reads an image from a file and preprocesses it and returns."""

		image_path = self.image_paths[index]
		# filename = image_path.split('_')[-1][:-len(".jpg")]
		filename = image_path[-17:]  # TODO: Change when using image patches because of naming convention: 17 for patch 9 for not patch
		GT_path = (self.GT_paths + filename)  # TODO: bug when crating images to name a space
		# print(GT_path)

		image = Image.open(image_path)
		GT = Image.open(GT_path)

		# for using Clahe
		image = cv2.imread(image_path)

		GT = cv2.imread(GT_path)

		# image
		lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

		lab_planes = cv2.split(lab)

		clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))

		lab_planes[0] = clahe.apply(lab_planes[0])

		lab = cv2.merge(lab_planes)

		image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
		# GT

		lab = cv2.cvtColor(GT, cv2.COLOR_BGR2LAB)

		lab_planes = cv2.split(lab)

		clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))

		lab_planes[0] = clahe.apply(lab_planes[0])

		lab = cv2.merge(lab_planes)

		GT = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

		# aspect_ratio = image.size[1]/image.size[0]
		# aspect_ratio = ima
		#

		#
		# ResizeRange = random.randint(300, 320)
		# # Transform.append(T.Resize((int(ResizeRange*aspect_ratio))))
		# p_transform = random.random()
		# #
		# if (self.mode == 'train') and p_transform <= self.augmentation_prob:
		# 	RotationDegree = 2  # random.randint(0, 3)
		# 	RotationDegree = self.RotationDegree[RotationDegree]
		# 	if (RotationDegree == 90) or (RotationDegree == 270):
		# 		aspect_ratio = 1/aspect_ratio
		# 	# print(RotationDegree)

		# 	# Transform.append(T.RandomRotation((RotationDegree)))  # Only 1 argument

		# 	RotationRange = 10  # random.randint(0,20)
		# 	Transform.append(T.RandomRotation((RotationRange)))  # Only 1 argument
		# 	CropRange = 48  # random.randint(250,270)
		# 	Transform.append(T.CenterCrop((int(CropRange*aspect_ratio))))
		# 	Transform = T.Compose(Transform)

		# 	image = Transform(image)
		# 	GT = Transform(GT)

		# 	ShiftRange_left = 10  # random.randint(0,20)
		# 	ShiftRange_upper = 10  # random.randint(0,20)
		# 	ShiftRange_right = image.size[0] - 10  # random.randint(0,20)
		# 	ShiftRange_lower = image.size[1] - 10  # random.randint(0,20)
		# 	image = image.crop(box=(ShiftRange_left,ShiftRange_upper,ShiftRange_right,ShiftRange_lower))
		# 	GT = GT.crop(box=(ShiftRange_left,ShiftRange_upper,ShiftRange_right,ShiftRange_lower))

		# 	if random.random() < 0.5:
		# 		image = TF.hflip(image)
		# 		GT = TF.hflip(GT)

		# 	if random.random() < 0.5:
		# 		image = TF.vflip(image)
		# 		GT = TF.vflip(GT)

		# 	# Transform = T.ColorJitter(brightness=0.2,contrast=0.2,hue=0.02)

		# 	image = Transform(image)

		Transform = []
		#
		# Transform.append(T.Resize((int(48*aspect_ratio)-int(48*aspect_ratio)%16)))
		Transform.append(T.ToTensor())
		Transform = T.Compose(Transform)
		image = Transform(image)
		Norm_ = T.Normalize((0.7810, 0.3966, 0.2238), (0.0361, 0.0341, 0.0180))  # This is for DRIVE
		image = Norm_(image)
		# print(image.shape)
		# TODO: green channel only
		image = image[1:2, :, :]
		# print(image)
		# print(image.shape)

		GT = Transform(GT)
		GT = GT[0:1, :, :]
		# print(GT.shape)

		#

		return image, GT, image_path

	def __len__(self):
		"""Returns the total number of font files."""
		return len(self.image_paths)


def get_loader(image_path, image_size, batch_size, num_workers=8, mode='train', augmentation_prob=0.4):
	"""Builds and returns Dataloader."""

	dataset = ImageFolder(root=image_path, image_size=image_size, mode=mode, augmentation_prob=augmentation_prob)
	data_loader = data.DataLoader(dataset=dataset,
								  batch_size=batch_size,
								  shuffle=True,
								  num_workers=num_workers)
	return data_loader
