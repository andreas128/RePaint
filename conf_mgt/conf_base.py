# Copyright (c) 2022 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This repository was forked from https://github.com/openai/guided-diffusion, which is under the MIT license

from functools import lru_cache
import os
import torch
from utils import imwrite

from collections import defaultdict
from os.path import isfile, expanduser

def to_file_ext(img_names, ext):
    img_names_out = []
    for img_name in img_names:
        splits = img_name.split('.')
        if not len(splits) == 2:
            raise RuntimeError("File name needs exactly one '.':", img_name)
        img_names_out.append(splits[0] + '.' + ext)

    return img_names_out

def write_images(imgs, img_names, dir_path):
    os.makedirs(dir_path, exist_ok=True)

    for image_name, image in zip(img_names, imgs):
        out_path = os.path.join(dir_path, image_name)
        imwrite(img=image, path=out_path)



class NoneDict(defaultdict):
    def __init__(self):
        super().__init__(self.return_None)

    @staticmethod
    def return_None():
        return None

    def __getattr__(self, attr):
        return self.get(attr)


class Default_Conf(NoneDict):
    def __init__(self):
        pass

    def get_dataloader(self, dset='train', dsName=None, batch_size=None, return_dataset=False):

        if batch_size is None:
            batch_size = self.batch_size

        candidates = self['data'][dset]
        ds_conf = candidates[dsName].copy()

        if ds_conf.get('mask_loader', False):
            from guided_diffusion.image_datasets import load_data_inpa
            return load_data_inpa(**ds_conf, conf=self)
        else:
            raise NotImplementedError()

    def get_debug_variance_path(self):
        return os.path.expanduser(os.path.join(self.get_default_eval_conf()['paths']['root'], 'debug/debug_variance'))

    @ staticmethod
    def device():
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    def eval_imswrite(self, srs=None, img_names=None, dset=None, name=None, ext='png', lrs=None, gts=None, gt_keep_masks=None, verify_same=True):
        img_names = to_file_ext(img_names, ext)

        if dset is None:
            dset = self.get_default_eval_name()

        max_len = self['data'][dset][name].get('max_len')

        if srs is not None:
            sr_dir_path = expanduser(self['data'][dset][name]['paths']['srs'])
            write_images(srs, img_names, sr_dir_path)

        if gt_keep_masks is not None:
            mask_dir_path = expanduser(
                self['data'][dset][name]['paths']['gt_keep_masks'])
            write_images(gt_keep_masks, img_names, mask_dir_path)

        gts_path = self['data'][dset][name]['paths'].get('gts')
        if gts is not None and gts_path:
            gt_dir_path = expanduser(gts_path)
            write_images(gts, img_names, gt_dir_path)

        if lrs is not None:
            lrs_dir_path = expanduser(
                self['data'][dset][name]['paths']['lrs'])
            write_images(lrs, img_names, lrs_dir_path)

    def get_default_eval_name(self):
        candidates = self['data']['eval'].keys()
        if len(candidates) != 1:
            raise RuntimeError(
                f"Need exactly one candidate for {self.name}: {candidates}")
        return list(candidates)[0]

    def pget(self, name, default=None):
        if '.' in name:
            names = name.split('.')
        else:
            names = [name]

        sub_dict = self
        for name in names:
            sub_dict = sub_dict.get(name, default)

            if sub_dict == None:
                return default

        return sub_dict
