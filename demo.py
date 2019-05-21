import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import json
import skimage.draw

ROOT_DIR = os.path.abspath('/')

# import Mask RCNN
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize

COCO_WEIGHTS_PATH = os.path.abspath('/home/henry/ai/tfProject/Mask_RCNN/mask_rcnn_coco.h5')  # TODO
OUTPUT_DIR = os.path.abspath("/home/henry/ai/KakaL/ep_m_rcnn/output")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(OUTPUT_DIR, 'logs')

# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results/")

# The dataset doesn't have a standard train/val split, so I picked
# a variety of images to surve as a validation set.
VAL_IMAGE_IDS = ["w01_Y_126",
                 "w01_Y_127",
                 "w01_Y_128",
                 "w01_Y_129",
                 "w01_Y_130",
                 "w01_Y_131",
                 "w01_Y_132",
                 "w01_Y_133",
                 "w01_Y_134",
                 "w01_Y_135",
                 "w01_Y_137",
                 "w01_Y_138",
                 "w01_Y_139",
                 "w01_R_30",
                 "w01_R_31",
                 "w01_R_32",
                 "w01_R_33",
                 "w01_R_34",
                 "w01_R_35",
                 "w01_R_36",
                 "w01_R_37",
                 "w01_R_38",
                 "w01_R_39",
                 "w01_R_40",
                 "w01_R_41",
                 "w01_R_42",
                 "w01_G_77",
                 "w01_G_78",
                 "w01_G_79",
                 "w01_G_80",
                 "w01_G_81",
                 "w01_G_82",
                 "w01_G_83",
                 "w01_G_84",
                 "w01_G_85",
                 ]


##################################################################
# Configurations
##################################################################


class LesionConfig(Config):
    """
    病灶检测的配置
    """
    NAME = 'Lesion'

    # BACKBONE = "resnet50"
    IMAGES_PER_GPU = 1

    NUM_CLASSES = 1+1  # todo 病灶 + 健康
    # 每个EPOCH训练步数
    STEPS_PER_EPOCH = 100
    # 最小得分，根据目的，为了提高召回率可以适当降低
    DETECTION_MIN_CONFIDENCE = 0.8  # todo

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 2

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 2

    # 修改为灰度图
    # IMAGE_CHANNEL_COUNT = 1
    # MEAN_PIXEL = np.array([123.7])


############################################################
#  Dataset
############################################################

class LesionDataset(utils.Dataset):

    def load_lesion(self, dataset_dir, subset):
        """
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """

        # Add classes. We have only one class to add.
        self.add_class("lesion", 1, "lesion")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        subset_dir = "train" if subset in ["train", "val"] else subset
        dataset_dir = os.path.join(dataset_dir, subset_dir)
        if subset == "val":
            image_ids = VAL_IMAGE_IDS
        else:
            # Get image ids from directory names
            images = next(os.walk(dataset_dir))[2]
            image_ids = []
            for image in images:
                image = os.path.splitext(image)[0]
                image_ids.append(image)
            if subset == "train":
                image_ids = list(set(image_ids) - set(VAL_IMAGE_IDS))

        # Add images
        for image_id in image_ids:
            self.add_image(
                "lesion",
                image_id=image_id,
                path=os.path.join(dataset_dir, "{}.jpg".format(image_id))
            )

    def load_mask(self, image_id):
        """
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        Mask_DIR = os.path.abspath('/home/henry/ai/dlDate/mrcnn_fcd/train_ann')
        image_name = self.image_info[image_id]['id']

        # Get mask directory from image path
        filename = image_name[:4]+'label_'+image_name[4:]
        mask_path = os.path.join(Mask_DIR, "{}.jpg".format(filename))
        mask = []
        m = skimage.io.imread(mask_path).astype(np.bool)
        mask.append(m)
        # mask = skimage.color.gray2rgb(skimage.io.imread(mask_path).astype(np.bool))
        mask = np.stack(mask, axis=-1)
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "lesion":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


############################################################
#  Training
############################################################

def train(model):
    """Train the model"""
    # Training dataset
    dataset_train = LesionDataset()
    dataset_train.load_lesion(args.dataset, 'train')
    dataset_train.prepare()

    # Validation dataset
    dataset_val = LesionDataset()
    dataset_val.load_lesion(args.dataset, "val")
    dataset_val.prepare()

    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')


############################################################
#  Detection
############################################################

def inference(model, image_path):
    print("Running on {}".format(image_path))
    image = skimage.color.gray2rgb(skimage.io.imread(image_path))
    # Run detection
    results = model.detect([image], verbose=1)
    # Visualize results
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 'lesion', r['scores'])


if __name__ == '__main__':
    import argparse

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Train Model to detect lesion.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'inference'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory of the lesion dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the inference effect on')
    args = parser.parse_args()

    # Validate arguments
    # assert 在后面的条件出错时报错
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "inference":
        assert args.image, "Provide --image to apply inference"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = LesionConfig()
    else:
        class InferenceConfig(LesionConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()


    # Creat Model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    else:
        weights_path = args.weights


    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "inference":
        inference(model, args.image)
        # detect_and_color_splash(model, image_path=args.image)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'inference'".format(args.command))



