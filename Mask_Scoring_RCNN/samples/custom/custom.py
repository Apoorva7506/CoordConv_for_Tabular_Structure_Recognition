"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
Edited for general application by Soumya Yadav (Psoumyadav@gmail.com)

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import logging
import numpy as np
import skimage.draw
import imgaug.augmenters as iaa
# Root directory of the project
ROOT_DIR = os.path.abspath("/content/drive/MyDrive/datasets/input")
PROJECT_DIR = os.path.abspath('/content/drive/MyDrive/Mask_Scoring_RCNN')

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
sys.path.append(PROJECT_DIR)
print(sys.path)
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "tuneic17")

############################################################
#  Configurations
############################################################


class CustomConfig(Config):
    NAME = "cell"
    # RPN_ANCHOR_RATIOS = [1.0, 2.32, 2.66, 3.77, 5.05, 12.06]
    #Below RATIOS FOR tab det icdar19 track A
    #RPN_ANCHOR_RATIOS =  [1,1,1,1,1]
    #RPN_ANCHOR_RATIOS =  [0.33, 0.73, 1.0, 1.34, 2.77]
    #RATIOS FOR ICDAR 17 BELOW
    #RPN_ANCHOR_RATIOS =  [0.63,1,1,1.07,1.23,1.64]
    #Ratios for icdar19
    RPN_ANCHOR_RATIOS=[1.94, 2.97, 4.26, 4.56, 5.34, 13.31]
    #TRACK A BELOW
    RPN_ANCHOR_SCALES=[16,32, 64, 128, 256]
    #POST_NMS_ROIS_INFERENCE = 1000
    #DETECTION_NMS_THRESHOLD=0.5
    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    
    IMAGES_PER_GPU = 1
    #DETECTION_NMS_THRESHOLD=0.8
    USE_MINI_MASK = True
    TRAIN_ROIS_PER_IMAGE=200
    MINI_MASK_SHAPE=(56,56)               
    # Number of classes (including background)
    # NUM_CLASSES = 1 + 2  # Background + number of classes (Here, 2)
    NUM_CLASSES = 1 + 1
    LEARNING_MOMENTUM= 0.9
    #IMAGE_RESIZE_MODE = "pad64"
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.78
    IMAGES_PER_GPU = 1
    #STEPS_PER_EPOCH = 100
    POST_NMS_ROIS_TRAINING = 200
    POST_NMS_ROIS_INFERENCE = 200
    #POST_NMS_ROIS_INFERENCE = 100
    DETECTION_MAX_INSTANCES = 200
    LEARNING_RATE = 0.00004
    RPN_NMS_THRESHOLD = 0.7
    MAX_GT_INSTANCES = 200
    TRAIN_BN=False

    # ---------------xxxxx----------------
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 180
    VALIDATION_STEPS = 85
#----------------------------------------------------------------- Old hyperparameters below
    # # TRAIN_ROIS_PER_IMAGE = 200
    # # Give the configuration a recognizable name
    # NAME = "cell"
    # #RPN_ANCHOR_RATIOS = [1.0, 2.32, 2.66, 3.77, 5.05, 12.06]
    # RPN_ANCHOR_RATIOS = [0.11, 0.25, 5.91, 12.58, 29.36]
    # LEARNING_RATE=0.00005
    # #IMAGE_CHANNEL_COUNT = 5
    # # We use a GPU with 12GB memory, which can fit two images.
    # # Adjust down if you use a smaller GPU.
    # IMAGES_PER_GPU = 1
    # DETECTION_MAX_INSTANCES =170      
    # DETECTION_MIN_CONFIDENCE =0.7
    # # Number of classes (including background)
    # # NUM_CLASSES = 1 + 2  # Background + number of classes (Here, 2)
    # NUM_CLASSES = 1 + 1
    
    # #IMAGE_CHANNEL_COUNT=5
    # TRAIN_ROIS_PER_IMAGE=200
    
    # # Skip detections with < 90% confidence
    # #DETECTION_MIN_CONFIDENCE = 0.9

    # # ---------------xxxxx----------------
    # # Number of training steps per epoch
    # # STEPS_PER_EPOCH = 100
    # STEPS_PER_EPOCH = 210
    # VALIDATION_STEPS = 105

    # # TRAIN_ROIS_PER_IMAGE = 200

############################################################
#  Dataset
############################################################

class CustomDataset(utils.Dataset):

    def load_custom(self, dataset_dir, subset):
        """Load a subset of the custom dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes according to the numbe of classes required to detect
        # self.add_class("custom", 1, "object1")
        # self.add_class("custom",2,"object2")
        
        # our cell
        self.add_class("custom", 1, "cell")

        # Train or validation dataset?
        assert subset in ["train", "val", 'test','jpg']
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.

        # our json file name
        # annotations = json.load(open(os.path.join(dataset_dir, "output.json")))

        # testing(other) json file name
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            polygons = [r['shape_attributes'] for r in a['regions'].values()]
            #labelling each class in the given image to a number

            # custom = [s['region_attributes'] for s in a['regions'].values()]
            
            num_ids=[]
            #Add the classes according to the requirement
            # for n in custom:
            #     try:
            #         if n['label']=='cell':
            #             num_ids.append(1)
            #         elif n['label']=='object2':
            #             num_ids.append(2)
            #     except:
            #         pass
            for _ in polygons:
                num_ids.append(1)

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "custom",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                num_ids=num_ids)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a custom dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        
        
        if image_info["source"] != "custom":
            return super(self.__class__, self).load_mask(image_id)
        num_ids = image_info['num_ids']	
        #print("Here is the numID",num_ids)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        num_ids = np.array(num_ids, dtype=np.int32)
        
        return mask, num_ids#.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32), 

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "custom":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = CustomDataset()
    dataset_train.load_custom(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CustomDataset()
    dataset_val.load_custom(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    augmentation = iaa.SomeOf((0, 3), [
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.OneOf([iaa.Affine(rotate=90),
               iaa.Affine(rotate=180),
               iaa.Affine(rotate=270)],
             ),
    iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}),
    iaa.Multiply((0.8, 1.5)),
    iaa.GaussianBlur(sigma=(0.0, 5.0))
])
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=100,
                #augmentation=augmentation,
                layers='all')

     
    # print("Training network heads")
    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE,
    #             #custom_callbacks = [tensorboard_callback],
    #             epochs=500,
    #             layers='all')


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect custom objects.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/custom/dataset/",
                        help='Directory of the custom dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)
    print("*********************\n\n")
    

    # Configurations
    if args.command == "train":
        config = CustomConfig()
    else:
        class InferenceConfig(CustomConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
        
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)
    print('----------------------------------The model is here below-----------------------------------------')
    print(model.keras_model.summary())
    print('--------------------------x-x-x-x-x----------------------------')

    # Select weights file to load
    if args.weights.lower() == "new":	
        print("weight path entered")	
        # print(NEW_WEIGHTS_PATH)	
        # weights_path = NEW_WEIGHTS_PATH
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True,exclude=["conv1","rpn_model","mrcnn_class_logits","mrcnn_bbox_fc","mrcnn_bbox", "mrcnn_mask"])
    else:

      l=["conv1","res5a_branch2a","res5b_branch2a","res5c_branch2a","res5d_branch2a","res5e_branch2a","res3c_branch2a","res4e_branch2a","res4a_branch2a","res4f_branch2a","res4d_branch2a","res4c_branch2a","res4b_branch2a","res3d_branch2a","res2a_branch2a","res3b_branch2a","res2b_branch2a","res2c_branch2a","res3a_branch2a","mrcnn_class_logits", "mrcnn_bbox_fc","mrcnn_bbox", "mrcnn_mask","rpn_model"]
      
      l1=["conv1","res2a_branch2a","res3a_branch2a","res4a_branch2a","res5a_branch2a","mrcnn_class_logits", "mrcnn_bbox_fc", 
                                  "mrcnn_bbox", "mrcnn_mask","rpn_model"]
      print("YOOOO99999")

      model.load_weights(weights_path, by_name=True, exclude=["rpn_model"])
    # Train or evaluate
    if args.command == "train":
        #print(model.summary())
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))

