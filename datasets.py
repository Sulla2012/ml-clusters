#A class for handling WISE data, safe to ignore
class WiseDatasetNeg(torch.utils.data.Dataset):
    def __init__(self, root, transforms, randoms = False, bands = 3):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned

        self.imgs = list(sorted(os.listdir(os.path.join(root, "comb"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "masks"))))
        self.randoms = randoms
        
        self.bands = bands
    def __getitem__(self, idx):
        # load images and masks

        img_path = os.path.join(self.root, "comb", self.imgs[idx])

        with np.load(img_path) as data:
            img = data['arr_0']
            img = np.array(img, dtype='f')
        
        if self.imgs[idx][0:7] == 'randoms':
            
            image_id = torch.tensor([idx])
                
            target = {}
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((1, ), dtype=torch.int64)
            target["image_id"] = image_id
            target["area"] = torch.zeros((0,), dtype=torch.int64)
            target["iscrowd"] = torch.zeros((0,), dtype=torch.int64)
            
            
        else:
            mask_path = os.path.join(self.root, "masks", "cluster_mask_{}.npz".format(self.imgs[idx][8:12]))
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
        
        img = img[..., self.bands]
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
