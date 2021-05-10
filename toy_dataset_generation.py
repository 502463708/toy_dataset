import copy
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mmcv
import os
import random
import shutil
import tqdm

from utils import *


class ToyDatasetGenerator:
    def __init__(self, num_expected_samples, **kwargs):
        self.num_expected_samples = num_expected_samples

        dataset_type = 'train'
        if 'dataset_type' in kwargs:
            dataset_type = kwargs['dataset_type']
        self.image_dir = 'images_{}/'.format(dataset_type)
        self.mask_dir = 'masks_{}/'.format(dataset_type)
        self.json_path = 'annotations/instances_{}.json'.format(dataset_type)

        if os.path.exists(self.image_dir):
            delete_flag = input('Directory {} already exists, would you like to delete it?'.format(self.image_dir))
            if delete_flag in ['y', 'yes', 'Y', 'YES', 'Yes']:
                shutil.rmtree(self.image_dir)
            else:
                raise IOError
        os.mkdir(self.image_dir)

        if os.path.exists(self.mask_dir):
            delete_flag = input('Directory {} already exists, would you like to delete it?'.format(self.mask_dir))
            if delete_flag in ['y', 'yes', 'Y', 'YES', 'Yes']:
                shutil.rmtree(self.mask_dir)
            else:
                raise IOError
        os.mkdir(self.mask_dir)

        self.instance_per_image = 1
        if 'instance_per_image' in kwargs:
            self.instance_per_image = kwargs['instance_per_image']

        self.dilation_range = (5, 20)
        if 'dilation_range' in kwargs:
            self.dilation_range = kwargs['dilation_range']

        self.shrink_range = (0.3, 0.6)
        if 'shrink_range' in kwargs:
            self.shrink_range = kwargs['shrink_range']

        self.skeleton_dir = 'skeletons/'
        if 'skeleton_dir' in kwargs:
            self.skeleton_dir = kwargs['skeleton_dir']
        self.skeletons = self.load_skeleton()

        self.obj_count = 0
        self.coco_format_json = dict(
            images=list(),
            annotations=list(),
            categories=[{'id': 0, 'name': 'curve'}])

        return

    def load_skeleton(self):
        assert os.path.exists(self.skeleton_dir)
        skeleton_filenames = os.listdir(self.skeleton_dir)
        assert len(skeleton_filenames) > 0
        # load skeletons
        skeletons = list()
        for skeleton_filename in skeleton_filenames:
            skeleton = cv2.imread(os.path.join(self.skeleton_dir, skeleton_filename), cv2.IMREAD_GRAYSCALE)
            skeleton = set_background_zero(skeleton)
            skeletons.append(skeleton)

        return skeletons

    def get_mask(self):
        # load skeleton
        skeleton = random.choice(self.skeletons)
        mask = copy.copy(skeleton)
        src_height, src_width = mask.shape

        # dilate
        radius = random.randint(self.dilation_range[0], self.dilation_range[1])
        mask = dilate_mask(mask, radius)

        # rotate
        angle = random.randint(0, 360)
        matrix_rotate = cv2.getRotationMatrix2D((src_height / 2.0, src_width / 2.0), angle, 1)
        mask = cv2.warpAffine(mask, matrix_rotate, (src_height, src_width), borderValue=0)

        # shrink
        shrink_factor_height = random.uniform(self.shrink_range[0], self.shrink_range[1])
        shrink_factor_width = random.uniform(self.shrink_range[0], self.shrink_range[1])
        mask = cv2.resize(mask, dsize=None, fx=shrink_factor_width, fy=shrink_factor_height,
                          interpolation=cv2.INTER_NEAREST)

        # padding (shift)
        cur_height, cur_width = mask.shape
        gap_y = src_height - cur_height
        gap_x = src_width - cur_width
        ratio_top = random.random()
        ratio_left = random.random()
        padding_top = int(ratio_top * gap_y)
        padding_bottom = gap_y - padding_top
        padding_left = int(ratio_left * gap_x)
        padding_right = gap_x - padding_left
        mask = cv2.copyMakeBorder(mask, padding_top, padding_bottom, padding_left, padding_right,
                                  cv2.BORDER_CONSTANT, value=0)
        mask[mask != 0] = 255

        mask = mask.astype(np.uint8)

        return mask

    def get_masks(self, num_samples):
        masks = list()

        for idx in range(num_samples):
            mask = self.get_mask()
            masks.append(mask)

        return masks

    def update_coco_image(self, image_idx, filename, masks):
        image_c1 = np.random.uniform(0, 80) + np.random.normal(loc=0, scale=10, size=masks[0].shape)
        image_c2 = np.random.uniform(0, 80) + np.random.normal(loc=0, scale=10, size=masks[0].shape)
        image_c3 = np.random.uniform(0, 80) + np.random.normal(loc=0, scale=10, size=masks[0].shape)

        for idx in range(len(masks)):
            mask = masks[idx]
            background_idxes = mask == 0

            foreground_1 = np.random.uniform(100, 200) + np.random.normal(loc=0, scale=10, size=masks[0].shape)
            foreground_2 = np.random.uniform(100, 200) + np.random.normal(loc=0, scale=10, size=masks[0].shape)
            foreground_3 = np.random.uniform(100, 200) + np.random.normal(loc=0, scale=10, size=masks[0].shape)

            image_c1[~background_idxes] = foreground_1[~background_idxes]
            image_c2[~background_idxes] = foreground_2[~background_idxes]
            image_c3[~background_idxes] = foreground_3[~background_idxes]

        image_list = [image_c1, image_c2, image_c3]
        image = cv2.merge(image_list)
        image = np.clip(image, 0, 255)
        image = image.astype(np.uint8)

        coco_image = dict(
            id=image_idx,
            file_name=filename,
            height=image.shape[0],
            width=image.shape[1])

        self.coco_format_json['images'].append(coco_image)

        return image

    def update_coco_anno(self, image_idx, masks):
        for idx_1 in range(len(masks)):
            complete_mask = copy.copy(masks[idx_1])
            y_min, y_max, x_min, x_max = get_bounding_box_from_mask(complete_mask)

            mask = copy.copy(complete_mask)
            for idx_2 in range(idx_1 + 1, len(masks)):
                mask_occlude = masks[idx_2]
                mask[mask_occlude != 0] = 0

            poly = mask2polygon(mask)

            data_anno = dict(
                image_id=image_idx,
                id=self.obj_count,
                category_id=0,
                bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                area=(mask != 0).sum(),
                segmentation=poly,
                iscrowd=0)

            self.coco_format_json['annotations'].append(data_anno)
            self.obj_count += 1

        return

    def run(self):
        for idx in tqdm.tqdm(range(self.num_expected_samples)):
            filename = '{}.png'.format(idx)
            masks = self.get_masks(self.instance_per_image)
            image = self.update_coco_image(idx, filename, masks)
            cv2.imwrite(os.path.join(self.image_dir, filename), image)
            self.update_coco_anno(idx, masks)

            for idx_2 in range(len(masks)):
                cv2.imwrite(os.path.join(self.mask_dir, filename.replace('.png', '_{}.png'.format(idx_2))),
                            masks[idx_2])

        mmcv.dump(self.coco_format_json, self.json_path)

        return


if __name__ == '__main__':
    a = ToyDatasetGenerator(num_expected_samples=1000, dataset_type='val', instance_per_image=10,
                            dilation_range=(5, 20), shrink_range=(0.3, 0.7))
    a.run()
