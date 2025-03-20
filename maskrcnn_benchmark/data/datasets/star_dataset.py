import os

import json
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import numpy as np
import re
import csv
from maskrcnn_benchmark.structures.bounding_box import BoxList
from collections import defaultdict
from tqdm import tqdm
import random
import mmcv
# from mmcv.ops import nms_rotated
import warnings
from functools import partial
from shapely.geometry import Polygon

class StarDataset(Dataset):
    def __init__(self, dataset_dir, transforms=None, box_filter_thresh=0.5, split="train",
                 num_im=-1, flip_aug=False, filter_empty_rels=True,
                 filter_duplicate_rels=True, filter_non_overlap=True,
                 custom_eval=False, custom_path=''):
        """
        Args:
            dataset_dir (str): Root directory of the dataset.
            transforms (callable, optional): A function/transform to apply to the images.
            box_filter_thresh (float): Threshold for filtering bounding boxes.
            split (str): Dataset split, "train" or "val".
            num_im (int): Number of images to load.
            num_val_im (int): Number of validation images.
            filter_empty_rels (bool): Whether to filter samples with no relationships.
            filter_non_overlap (bool): Whether to filter relationships with no bounding box overlap.
        """
        num_im = 100
        self.split = split
        self.dataset_dir = dataset_dir
        self.image_dir = os.path.join(dataset_dir, f"{split}/img")
        self.annotation_dir = os.path.join(dataset_dir, f"{split}/object-json")
        self.relation_dir = os.path.join(dataset_dir, f"{split}/relationship")
        self.info_file = os.path.join(dataset_dir, f"star_info.json")
        img_info_json_file = os.path.join(dataset_dir, f"{split}/{split}_img_info.json")

        self.transforms = transforms
        self.flip_aug = flip_aug
        self.box_filter_thresh = box_filter_thresh
        self.filter_non_overlap = filter_non_overlap and self.split == 'train'
        self.filter_duplicate_rels = filter_duplicate_rels and self.split == 'train'

        # Load annotations and graphs
        self.ind_to_classes, self.classes_to_ind, self.ind_to_predicates, self.predicates_to_ind \
            = load_info(self.info_file)


        if num_im > 0:
            img_list = [img_name for img_name in sorted(os.listdir(self.image_dir))[:num_im]]
            self.gt_mask = np.array([True] * num_im + [False] * (len(os.listdir(self.image_dir))-num_im))
        else:
            img_list = [img_name for img_name in sorted(os.listdir(self.image_dir))]
            self.gt_mask = np.array([True] * len(os.listdir(self.image_dir)))
        self.gt_images, self.gt_boxes, self.gt_classes, self.gt_relationships = self.load_graphs(
            img_list,
            self.annotation_dir,
            self.relation_dir,
        )
        self.image_ids = list(range(len(self.gt_images)))

        self.filenames, self.img_info = load_image_filenames(self.image_dir, img_info_json_file)  # length equals to split_mask
        # 筛选出用于训练的
        self.filenames = [self.filenames[i] for i in np.where(self.gt_mask)[0]]
        self.img_info = [self.img_info[i] for i in np.where(self.gt_mask)[0]]

    def load_graphs(self, image_list, annotation_dir, relation_dir):
        """
        Load graph-related data from annotations.
        """
        gt_images = []
        gt_boxes = []
        gt_classes = []
        gt_relations = []

        for idx, filename in enumerate(image_list):
            file_path_anno = os.path.join(annotation_dir, filename[:-4]) + ".json"
            file_path_rel = os.path.join(relation_dir, filename[:-4]) + ".csv"

            with open(file_path_anno, 'r') as f:
                data = json.load(f, object_hook=lambda d: {k: v for k, v in d.items() if k != 'imageData'})

            with open(file_path_rel, 'r') as file:
                reader = csv.DictReader(file)
                is_rel_file_empty = True  # 假设文件为空
                for _ in reader:
                    is_rel_file_empty = False  # 如果读取到任何行，文件不为空
                    break  # 读取到一行后即可退出循环
                file.seek(0)

            if is_rel_file_empty:
                continue

            if len(data['shapes']) == 0:
                continue

            sample_boxes = []
            sample_labels = []
            sample_groups = []
            sample_relations = []

            # TODO:要变成旋转框标注
            for shape in data.get('shapes', []):
                points = shape.get('points', [])
                label = self.classes_to_ind[shape.get('label', [])]
                group = shape.get('group_id', [])
                # 转换为 [x_min, y_min, x_max, y_max] 格式
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                x_min, y_min, x_max, y_max = min(x_coords), min(y_coords), max(x_coords), max(y_coords)
                sample_boxes.append([x_min, y_min, x_max, y_max])
                sample_labels.append(label)
                sample_groups.append(group)
            sample_groups = np.array(sample_groups)
            sample_labels = np.array(sample_labels)
            with open(file_path_rel, 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    relationship = self.predicates_to_ind.get(row['relationship'], 'unknown')
                    sub_group_ids = int(row['subject_ID'])
                    tar_group_ids = int(row['object_ID'])
                    assert relationship != 'unknown'
                    sub_ids = np.where(sample_groups == sub_group_ids)[0]
                    tar_ids = np.where(sample_groups == tar_group_ids)[0]
                    for sub_id in sub_ids:
                        for tar_id in tar_ids:
                            if self.classes_to_ind[row['subject']]==sample_labels[sub_id] and self.classes_to_ind[row['object']]==sample_labels[tar_id]:
                                sample_relations.append([sub_id, tar_id, relationship])

            if len(sample_relations)==0:
                print("erro")
            if len(sample_relations[0])==0:
                print("erro")

            gt_images.append(filename)
            gt_boxes.append(np.array(sample_boxes))  # 转换为 ndarray 格式
            gt_classes.append(np.array(sample_labels))
            gt_relations.append(np.array(sample_relations))

        return gt_images, gt_boxes, gt_classes, gt_relations

    def get_groundtruth(self, index, evaluation=False, flip_img=False):
        w, h = self.img_info[index]['width'], self.img_info[index]['height']
        # important: recover original box from BOX_SCALE
        box = self.gt_boxes[index]
        box = torch.from_numpy(box).reshape(-1, 4)  # guard against no boxes
        if flip_img:
            new_xmin = w - box[:, 2]
            new_xmax = w - box[:, 0]
            box[:, 0] = new_xmin
            box[:, 2] = new_xmax

        target = BoxList(box, (w, h), img_name=self.gt_images[index], mode='xyxy')  # xyxy
        target.add_field("labels", torch.from_numpy(self.gt_classes[index]))
        # TODO:应该是用不到attributes，先默认和class相同
        target.add_field("attributes", torch.from_numpy(self.gt_classes[index]))

        relation = self.gt_relationships[index].copy()  # (num_rel, 3)
        if self.filter_duplicate_rels:
            # Filter out dupes!
            assert self.split == 'train'
            old_size = relation.shape[0]
            all_rel_sets = defaultdict(list)
            for (o0, o1, r) in relation:
                all_rel_sets[(o0, o1)].append(r)
            relation = [(k[0], k[1], np.random.choice(v)) for k, v in all_rel_sets.items()]
            relation = np.array(relation, dtype=np.int32)

        # add relation to target
        num_box = len(target)
        relation_map = torch.zeros((num_box, num_box), dtype=torch.int64)
        for i in range(relation.shape[0]):
            if relation_map[int(relation[i, 0]), int(relation[i, 1])] > 0:
                if (random.random() > 0.5):
                    relation_map[int(relation[i, 0]), int(relation[i, 1])] = int(relation[i, 2])
            else:
                relation_map[int(relation[i, 0]), int(relation[i, 1])] = int(relation[i, 2])
        target.add_field("relation", relation_map, is_triplet=True)


        # # for debug
        # filename = os.path.basename(self.gt_images[index])  # 0000__1024__0___0.png
        # numbers = re.findall(r'\d+', filename)  # ['0000', '1024', '2472', '0']
        # numbers = list(map(int, numbers))  # 转换为整数
        # print(numbers)
        # metadata_tensor = torch.tensor(numbers, dtype=torch.float32)
        # target.add_field("img_name", metadata_tensor)

        if evaluation:
            target = target.clip_to_image(remove_empty=False)
            target.add_field("relation_tuple", torch.LongTensor(relation))  # for evaluation
            target.img_name = self.gt_images[index]
            return target
        else:
            target = target.clip_to_image(remove_empty=True)
            target.img_name = self.gt_images[index]
            return target

    def get_img_info(self, index):
        # WARNING: original image_file.json has several pictures with false image size
        # use correct function to check the validity before training
        # it will take a while, you only need to do it once

        # correct_img_info(self.img_dir, self.image_file)
        return self.img_info[index]

    def filter_non_overlap_rels(self, rels, boxes):
        """
        Filter relationships to keep only overlapping ones.
        """
        keep_rels = []
        for rel in rels:
            subj_idx, obj_idx = rel[0], rel[1]
            subj_box = boxes[subj_idx]
            obj_box = boxes[obj_idx]

            # Check overlap
            if self.check_overlap(subj_box, obj_box):
                keep_rels.append(rel)

        return torch.tensor(keep_rels, dtype=torch.long)

    def check_overlap(self, box1, box2):
        """
        Check if two boxes overlap.
        """
        x_min1, y_min1, x_max1, y_max1 = box1
        x_min2, y_min2, x_max2, y_max2 = box2

        return not (x_max1 < x_min2 or x_min1 > x_max2 or y_max1 < y_min2 or y_min1 > y_max2)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # Get image ID and its annotations
        img_name = self.gt_images[idx]
        # Load image
        img_path = os.path.join(self.image_dir, img_name)
        img = Image.open(img_path).convert('RGB')

        flip_img = (random.random() > 0.5) and self.flip_aug and (self.split == 'train')

        target = self.get_groundtruth(idx, flip_img)

        if flip_img:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, idx

    def filter_boxes(self, boxes, labels):
        """Filter bounding boxes based on size criteria."""
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]

        keep = (widths > self.box_filter_thresh) & (heights > self.box_filter_thresh)

        return boxes[keep], labels[keep]

    def get_statistics(self):
        fg_matrix, bg_matrix = get_STAR_statistics(self.dataset_dir, must_overlap=True)
        eps = 1e-3
        bg_matrix += 1
        fg_matrix[:, :, 0] = bg_matrix
        pred_dist = np.log(fg_matrix / fg_matrix.sum(2)[:, :, None] + eps)

        result = {
            'fg_matrix': torch.from_numpy(fg_matrix),
            'pred_dist': torch.from_numpy(pred_dist).float(),
            'obj_classes': self.ind_to_classes,
            'rel_classes': self.ind_to_predicates,
            'att_classes': self.ind_to_classes,
        }
        return result

    def merge_det(self, results, nproc=4):
        """Merging patch bboxes into full image.

        Args:
            results (list): Testing results of the dataset.
            nproc (int): number of process. Default: 4.

        Returns:
            list: merged results.
        """

        def extract_xy(img_id):
            """Extract x and y coordinates from image ID.

            Args:
                img_id (str): ID of the image.

            Returns:
                Tuple of two integers, the x and y coordinates.
            """
            pattern = re.compile(r'__(\d+)___(\d+)')
            match = pattern.search(img_id)
            if match:
                x, y = int(match.group(1)), int(match.group(2))
                return x, y
            else:
                warnings.warn(
                    "Can't find coordinates in filename, "
                    'the coordinates will be set to (0,0) by default.',
                    category=Warning)
                return 0, 0

        collector = defaultdict(list)
        for idx, img_id in enumerate(self.img_ids):
            result = results[idx]
            oriname = img_id.split('__', maxsplit=1)[0]
            x, y = extract_xy(img_id)
            new_result = []
            for i, dets in enumerate(result):
                bboxes, scores = dets[:, :-1], dets[:, [-1]]
                ori_bboxes = bboxes.copy()
                ori_bboxes[..., :2] = ori_bboxes[..., :2] + np.array(
                    [x, y], dtype=np.float32)
                labels = np.zeros((bboxes.shape[0], 1)) + i
                new_result.append(
                    np.concatenate([labels, ori_bboxes, scores], axis=1))
            new_result = np.concatenate(new_result, axis=0)
            collector[oriname].append(new_result)

        merge_func = partial(_merge_func, CLASSES=self.CLASSES, iou_thr=0.1)
        if nproc <= 1:
            print('Executing on Single Processor')
            merged_results = mmcv.track_iter_progress(
                (map(merge_func, collector.items()), len(collector)))
        else:
            print(f'Executing on {nproc} processors')
            merged_results = mmcv.track_parallel_progress(
                merge_func, list(collector.items()), nproc)

        # Return a zipped list of merged results
        return zip(*merged_results)



def get_STAR_statistics(dataset_dir, must_overlap=True):
    train_data = StarDataset(dataset_dir, num_im=-1, split='train',
                             filter_duplicate_rels=False)

    num_obj_classes = len(train_data.ind_to_classes)
    num_rel_classes = len(train_data.ind_to_predicates)
    fg_matrix = np.zeros((num_obj_classes, num_obj_classes, num_rel_classes), dtype=np.int64)
    bg_matrix = np.zeros((num_obj_classes, num_obj_classes), dtype=np.int64)

    for ex_ind in tqdm(range(len(train_data))):
        gt_classes = train_data.gt_classes[ex_ind].copy()
        gt_relations = train_data.gt_relationships[ex_ind].copy()
        gt_boxes = train_data.gt_boxes[ex_ind].copy()

        # For the foreground, we'll just look at everything
        o1o2 = gt_classes[gt_relations[:, :2]]
        for (o1, o2), gtr in zip(o1o2, gt_relations[:, 2]):
            fg_matrix[o1, o2, gtr] += 1

        # For the background, get all of the things that overlap.
        o1o2_total = gt_classes[np.array(
            box_filter(gt_boxes, must_overlap=must_overlap), dtype=int)]
        for (o1, o2) in o1o2_total:
            bg_matrix[o1, o2] += 1

    return fg_matrix, bg_matrix


def box_filter(boxes, must_overlap=False):
    """ Only include boxes that overlap as possible relations.
    If no overlapping boxes, use all of them."""
    n_cands = boxes.shape[0]

    overlaps = bbox_overlaps(boxes.astype(np.float64), boxes.astype(np.float64), to_move=0) > 0
    np.fill_diagonal(overlaps, 0)

    all_possib = np.ones_like(overlaps, dtype=bool)
    np.fill_diagonal(all_possib, 0)

    if must_overlap:
        possible_boxes = np.column_stack(np.where(overlaps))

        if possible_boxes.size == 0:
            possible_boxes = np.column_stack(np.where(all_possib))
    else:
        possible_boxes = np.column_stack(np.where(all_possib))
    return possible_boxes


def bbox_overlaps(boxes1, boxes2, to_move=1):
    """
    boxes1 : numpy, [num_obj, 4] (x1,y1,x2,y2)
    boxes2 : numpy, [num_obj, 4] (x1,y1,x2,y2)
    """
    #print('boxes1: ', boxes1.shape)
    #print('boxes2: ', boxes2.shape)
    num_box1 = boxes1.shape[0]
    num_box2 = boxes2.shape[0]
    lt = np.maximum(boxes1.reshape([num_box1, 1, -1])[:,:,:2], boxes2.reshape([1, num_box2, -1])[:,:,:2]) # [N,M,2]
    rb = np.minimum(boxes1.reshape([num_box1, 1, -1])[:,:,2:], boxes2.reshape([1, num_box2, -1])[:,:,2:]) # [N,M,2]

    wh = (rb - lt + to_move).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    return inter


def load_info(dict_file, add_bg=True):
    """
    Loads the file containing the visual genome label meanings
    """
    info = json.load(open(dict_file, 'r'))
    if add_bg:
        info['label_to_idx']['__background__'] = 0
        info['predicate_to_idx']['__background__'] = 0

    class_to_ind = info['label_to_idx']
    predicate_to_ind = info['predicate_to_idx']
    ind_to_classes = sorted(class_to_ind, key=lambda k: class_to_ind[k])
    ind_to_predicates = sorted(predicate_to_ind, key=lambda k: predicate_to_ind[k])

    return ind_to_classes, class_to_ind, ind_to_predicates, predicate_to_ind


def load_image_filenames(image_dir, img_info_json_file):
    # 直接从 JSON 文件中加载数据
    with open(img_info_json_file, 'r') as f:
        data = json.load(f)
        fns = data['filenames']
        img_info = data['image_info']

    # 构建完整的图像路径
    full_paths = [os.path.join(image_dir, fn.split('/')[-1]) for fn in fns]
    img_info = [{**img, 'imagePath': os.path.join(image_dir, img['imagePath'].split('/')[-1])} for img in img_info]

    # 检查文件是否存在
    valid_fns = [fp for fp in full_paths if os.path.exists(fp)]
    valid_img_info = [img for img in img_info if os.path.exists(img['imagePath'])]

    img_info_rename = [{'path': img['imagePath'], 'height': img['imageHeight'], 'width': img['imageWidth']}
                       for img in valid_img_info]

    assert len(valid_fns) == len(valid_img_info)
    return valid_fns, img_info_rename

def _merge_func(info, CLASSES, iou_thr):
    """Merging patch bboxes into full image.

    Args:
        CLASSES (list): Label category.
        iou_thr (float): Threshold of IoU.
    """

    def calculate_iou_rotated(box1, box2):
        """
        Calculate IoU for rotated bounding boxes using polygon intersection.

        Args:
            box1 (Tensor): First rotated box (x, y, w, h, angle).
            box2 (Tensor): Second rotated box (x, y, w, h, angle).

        Returns:
            float: IoU value between the two rotated boxes.
        """
        # Box format: (x, y, width, height, angle)
        def rotated_box_to_polygon(box):
            """
            Convert a rotated bounding box to a Shapely Polygon.

            Args:
                box (Tensor): Rotated box in (x, y, width, height, angle) format.

            Returns:
                Polygon: Shapely polygon representation of the rotated box.
            """
            x, y, w, h, angle = box
            # Create a rectangle around the center (x, y) with width w, height h, and rotate by angle
            rect = Polygon([(x - w / 2, y - h / 2),
                            (x + w / 2, y - h / 2),
                            (x + w / 2, y + h / 2),
                            (x - w / 2, y + h / 2)])
            # Rotate the rectangle by the angle
            rotated_rect = rect.rotate(angle, origin=(x, y))

            return rotated_rect
        # Convert rotated boxes to polygons
        poly1 = rotated_box_to_polygon(box1)
        poly2 = rotated_box_to_polygon(box2)

        # Calculate intersection area using polygon intersection
        intersection = poly1.intersection(poly2).area
        # Calculate the union area of the two polygons
        union = poly1.area + poly2.area - intersection

        # Return the IoU
        return intersection / union if union > 0 else 0.0

    def nms_rotated(bboxes, scores, iou_thr):
        """
        Custom NMS for rotated bounding boxes.

        Args:
            bboxes (Tensor): Rotated bounding boxes (N, 5), where each box is (x, y, w, h, angle).
            scores (Tensor): Tensor of shape (N,) for box scores.
            iou_thr (float): IoU threshold for NMS.

        Returns:
            keep_inds (Tensor): Indices of the boxes that are kept.
        """
        keep_inds = []
        sorted_indices = torch.argsort(scores, descending=True)
        while sorted_indices.numel() > 0:
            top_idx = sorted_indices[0]
            keep_inds.append(top_idx)
            sorted_indices = sorted_indices[1:]

            remaining_bboxes = bboxes[sorted_indices]
            remaining_scores = scores[sorted_indices]
            iou_scores = torch.tensor([calculate_iou_rotated(bboxes[top_idx], box) for box in remaining_bboxes])

            # Remove boxes with IoU greater than the threshold
            sorted_indices = sorted_indices[iou_scores < iou_thr]

        return torch.tensor(keep_inds)

    img_id, label_dets = info
    label_dets = np.concatenate(label_dets, axis=0)

    labels, dets = label_dets[:, 0], label_dets[:, 1:]

    big_img_results = []
    for i in range(len(CLASSES)):
        if len(dets[labels == i]) == 0:
            big_img_results.append(dets[labels == i])
        else:
            try:
                cls_dets = torch.from_numpy(dets[labels == i]).cuda()
            except:  # noqa: E722
                cls_dets = torch.from_numpy(dets[labels == i])
            nms_dets, keep_inds = nms_rotated(cls_dets[:, :5], cls_dets[:, -1],
                                              iou_thr)
            big_img_results.append(nms_dets.cpu().numpy())
    return img_id, big_img_results



# Example usage
if __name__ == "__main__":
    # dataset_dir = '/data/liuyi/dataset/relation_DATASETS/STAR-relationship/'
    dataset_dir = '/datasets/liuyi/STAR_split_data/split_ss_dota'
    from maskrcnn_benchmark.data.transforms import build_transforms
    from maskrcnn_benchmark.config import cfg
    import argparse
    from maskrcnn_benchmark.utils.comm import synchronize
    from maskrcnn_benchmark.data import make_data_loader

    parser = argparse.ArgumentParser(description="PyTorch Relation Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    debug_mode = True
    cfg.DATALOADER.NUM_WORKERS = 0 if debug_mode else 4
    cfg.freeze()

    is_train = True
    transforms = build_transforms(cfg, is_train)

    train_data_loader = make_data_loader(
        cfg,
        mode='train',
        is_distributed=False,
        start_iter=0,
    )

    for images, targets, id in train_data_loader:
        print("Images shape:", images.tensors.shape)
        print("Targets:", targets[0])
