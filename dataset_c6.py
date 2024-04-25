from dataset import ImageDataset
import os
from util.file_io import get_file_path_list
import cv2
from typing import Dict
import numpy as np
import torch
from torch import Tensor
import random
from imgproc import image_to_tensor


class ImageDatasetC6(ImageDataset):
    def __init__(
            self,
            src_images_dir: str,
            dst_images_dir: str,
            unpaired: bool,
            resized_image_size: int,
    ) -> None:
        self.src_dir=src_images_dir
        self.dst_dir=dst_images_dir
        self.src_skin_path_list=self.__get_image_list(src_images_dir)
        self.dst_skin_path_list=self.__get_image_list(dst_images_dir)
        self.unpaired = unpaired
        self.resized_image_size = resized_image_size

    def __get_image_list(self,path):
        file_list=get_file_path_list(path)
        img_list=[]
        for item in file_list:
            if item.endswith('_skin.jpg'):
                img_list.append(item)
        return img_list

    def __getitem__(self, batch_index: int) -> [Dict[str, Tensor], Dict[str, Tensor]]:
        # Read a batch of image data
        #print(self.src_skin_path_list[batch_index])

        src_skin_image = cv2.imread(self.src_skin_path_list[batch_index])
        src_dp_path = os.path.join(self.src_dir, os.path.basename(self.src_skin_path_list[batch_index]).split('_')[0]+'_dp.jpg')
        src_dp_image = cv2.imread(src_dp_path)
        if self.unpaired:
            dst_id=random.randint(0, len(self.src_skin_path_list) - 1)

        else:
            dst_id=batch_index
        dst_skin_image = cv2.imread(self.dst_skin_path_list[dst_id])
        dst_dp_path = os.path.join(self.dst_dir, os.path.basename(self.dst_skin_path_list[dst_id]).split('_')[0]+'_dp.jpg')
        dst_dp_image=cv2.imread(dst_dp_path)

        # Normalize the image data
        src_skin_image=src_skin_image.astype(np.float32) / 255.
        src_dp_image = src_dp_image.astype(np.float32) / 255.
        dst_skin_image = dst_skin_image.astype(np.float32) / 255.
        dst_dp_image = dst_dp_image.astype(np.float32) / 255.
        #src_image = src_image.astype(np.float32) / 255.
        #dst_image = dst_image.astype(np.float32) / 255.

        # Resized image
        src_skin_image = cv2.resize(src_skin_image, (self.resized_image_size, self.resized_image_size), interpolation=cv2.INTER_CUBIC)
        src_dp_image = cv2.resize(src_dp_image, (self.resized_image_size, self.resized_image_size),
                                    interpolation=cv2.INTER_CUBIC)
        dst_skin_image = cv2.resize(dst_skin_image, (self.resized_image_size, self.resized_image_size), interpolation=cv2.INTER_CUBIC)
        dst_dp_image = cv2.resize(dst_dp_image, (self.resized_image_size, self.resized_image_size), interpolation=cv2.INTER_CUBIC)


        # BGR convert RGB
        src_skin_image = cv2.cvtColor(src_skin_image, cv2.COLOR_BGR2RGB)
        src_dp_image = cv2.cvtColor(src_dp_image, cv2.COLOR_BGR2RGB)
        dst_skin_image = cv2.cvtColor(dst_skin_image, cv2.COLOR_BGR2RGB)
        dst_dp_image = cv2.cvtColor(dst_dp_image, cv2.COLOR_BGR2RGB)

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [-1, 1]
        src_skin_tensor = image_to_tensor(src_skin_image, True, False)
        src_dp_tensor = image_to_tensor(src_dp_image, True, False)
        src_tensor=torch.cat([src_skin_tensor,src_dp_tensor],0)
        dst_skin_tensor = image_to_tensor(dst_skin_image, True, False)
        dst_dp_tensor = image_to_tensor(dst_dp_image, True, False)
        dst_tensor=torch.cat([dst_skin_tensor,dst_dp_tensor],0)

        return {"src": src_tensor, "dst": dst_tensor}

    def __len__(self) -> int:
        return len(self.src_skin_path_list)
