import os
import numpy as np
import torch
from PIL import Image

from torch import nn, optim

import visionutils.transforms as T
from visionutils.engine import train_one_epoch, evaluate
import visionutils.utils

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from torchvision.models import resnet18, ResNet18_Weights

class ClusterDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms, cluster_dir, mask_dir):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        
        self.cluster_dir = cluster_dir
        self.mask_dir = mask_dir
        
        self.imgs = list(sorted(os.listdir(os.path.join(root, self.cluster_dir))))
        self.masks = list(sorted(os.listdir(os.path.join(root, self.mask_dir))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, self.cluster_dir, self.imgs[idx])
        
        mask_path = os.path.join(self.root, self.mask_dir, "{}_mask.npz".format(self.imgs[idx].strip('.npz')))
        #mask_path = os.path.join(self.root, self.mask_dir, self.masks[idx])

        #img = Image.open(img_path).convert("RGB")
        
        with np.load(img_path) as data:
            img = data['arr_0']
        img = np.array(img, dtype='f')

        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        with np.load(mask_path) as data:
            mask = data['arr_0']
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        
        if len(obj_ids) == 1: #If only background i.e. no objects
            
            #Return "Empty target"
            image_id = torch.tensor([idx])
                
            target = {}
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((1, ), dtype=torch.int64)
            target["image_id"] = image_id
            target["area"] = torch.zeros((0,), dtype=torch.int64)
            target["iscrowd"] = torch.zeros((0,), dtype=torch.int64)
        
        else: 
            # first id is the background, so remove it
            obj_ids = obj_ids[1:]
            # split the color-encoded mask into a set
            # of binary masks
            masks = mask == obj_ids[:, None, None]
            # get bounding box coordinates for each mask
            num_objs = len(obj_ids)
            boxes = []
            for i in range(num_objs):
                pos = np.where(masks[i])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                boxes.append([xmin, ymin, xmax, ymax])

            # convert everything into a torch.Tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # there is only one class
            labels = torch.ones((num_objs,), dtype=torch.int64)
            masks = torch.as_tensor(masks, dtype=torch.uint8)

            image_id = torch.tensor([idx])
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            # suppose all instances are not crowd
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["masks"] = masks
            target["image_id"] = image_id
            target["area"] = area
            target["iscrowd"] = iscrowd
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)



def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

def get_instance_frcnn_model(num_classes, backbone_type = 'mobilenet',
                             backbone_path = "/project/r/rbond/jorlo/ml-clusters/models/torch-act/act-mobilenet.pth"):
    if backbone_type == 'mobilenet':
        backbone_model = torchvision.models.mobilenet_v2()
        backbone_model.fc = nn.Linear(512, 2)

        if backbone_path is not None:
            backbone_model.load_state_dict(torch.load(backbone_path))
        backbone = backbone_model.features

        backbone.out_channels = 1280

        # let's make the RPN generate 5 x 3 anchors per spatial
        # location, with 5 different sizes and 3 different aspect
        # ratios. We have a Tuple[Tuple[int]] because each feature
        # map could potentially have different sizes and
        # aspect ratios
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                           aspect_ratios=((0.5, 1.0, 2.0),))

        # let's define what are the feature maps that we will
        # use to perform the region of interest cropping, as well as
        # the size of the crop after rescaling.
        # if your backbone returns a Tensor, featmap_names is expected to
        # be [0]. More generally, the backbone should return an
        # OrderedDict[Tensor], and in featmap_names you can choose which
        # feature maps to use.
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                        output_size=7,
                                                        sampling_ratio=2)


        model = FasterRCNN(backbone,
                       num_classes=2,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler,)
                       #box_score_thresh=0.9)

    elif backbone_type == 'resnet':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")

        # replace the classifier with a new one, that has
        # num_classes which is user-defined
        num_classes = num_classes
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)





    return model

###################################################################################################################################################################################################
im_type = 'indv'
root = '/project/r/rbond/jorlo/datasets/ACT_tiles/'
# use our dataset and defined transformations
dataset = ClusterDataset('/project/r/rbond/jorlo/datasets/ACT_tiles/', get_transform(train=True),
                        cluster_dir = 'indv_freq_tiles', mask_dir = 'indv_masks')
dataset_test = ClusterDataset('/project/r/rbond/jorlo/datasets/ACT_tiles/', get_transform(train=False),
                        cluster_dir = 'indv_freq_tiles', mask_dir = 'indv_masks')


#dataset = ClusterDataset('/project/r/rbond/jorlo/datasets/ACT_tiles/', get_transform(train=True),
#                        cluster_dir = 'small_freq_tiles', mask_dir = 'small_masks')
#dataset_test = ClusterDataset('/project/r/rbond/jorlo/datasets/ACT_tiles/', get_transform(train=False),
#                        cluster_dir = 'small_freq_tiles', mask_dir = 'small_masks')

# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()

test_num = 200
dataset = torch.utils.data.Subset(dataset, indices[:-test_num])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-test_num:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=4, shuffle=True, num_workers=4,
    collate_fn=visionutils.utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=visionutils.utils.collate_fn)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



# our dataset has two classes only - background and person
num_classes = 2

backbone = 'resnet'

# get the model using our helper function
model = get_instance_frcnn_model(num_classes, backbone_type = backbone)
#model = get_instance_frcnn_model(num_classes)

# move model to the right device

#torch.distributed.init_process_group(backend='nccl', world_size=4, rank=0)
#parallel_model = torch.nn.parallel.DistributedDataParallel(model, None)

model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

# let's train it for 10 epochs
from torch.optim.lr_scheduler import StepLR
num_epochs = 10

if os.path.exists('"/scratch/r/rbond/jorlo/ml-clusters/models/torch-act/act-{}-frcnn-indv.pth".format(backbone))'):
    model.load_state_dict(torch.load("/scratch/r/rbond/jorlo/ml-clusters/models/torch-act/act-{}-frcnn-indv.pth".format(backbone)))

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)
    torch.save(model.state_dict(), "/scratch/r/rbond/jorlo/ml-clusters/models/torch-act/act-{}-frcnn-indv.pth".format(backbone))



