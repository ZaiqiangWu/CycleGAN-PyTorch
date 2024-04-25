# Copyright 2022 Lorna Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse

import torch
from torchvision.utils import save_image

import model
from imgproc import preprocess_one_image
from utils import load_pretrained_state_dict
from util.file_io import get_file_path_list
import os


def main(args):
    device = torch.device(args.device)
    g_model = model.__dict__[args.model_arch_name](in_channels=3, out_channels=3, channels=64)
    g_model = g_model.to(device)

    # Load image
    img_path_list = get_file_path_list(args.inputs_path)


    # Load model weights
    g_model = load_pretrained_state_dict(g_model, False, args.model_weights_path)
    g_model.eval()

    for i in range(len(img_path_list)):
        image = preprocess_one_image(img_path_list[i], True, args.half, device)
        with torch.no_grad():
            gen_image = g_model(image)
            frame=torch.cat([image,gen_image.detach()],3)
            file_name=str(i).zfill(4)+'.jpg'
            save_image(frame, os.path.join(args.output_path,file_name), normalize=True)
            print(i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs_path", type=str, default="./data/skin_inpaint_cyclegan/testA",
                        help="Input image path. Default: ``./figure/``")
    parser.add_argument("--output_path", type=str, default="./figure/",
                        help="Output image path. Default: ``./figure/``")
    parser.add_argument("--model_arch_name", type=str, default="cyclenet",
                        help="Generator arch model name.  Default: ``cyclenet``")
    parser.add_argument("--model_weights_path", type=str,
                        default="./results/CycleGAN-skin_inpaint/g_B_best.pth.tar",
                        help="Generator model weights path.  Default: ``./results/pretrained_models/CycleGAN-apple2orange.pth.tar``")
    parser.add_argument("--half", action="store_true", default=False,
                        help="Use half precision. Default: ``False``")
    parser.add_argument("--device", type=str, default="cuda:0", choices=["cpu", 'cuda:0'],
                        help="Device. Default: ``cpu``")
    args = parser.parse_args()

    main(args)
