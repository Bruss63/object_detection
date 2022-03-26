import os
import random
import time
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.datasets.cityscapes as cityscpapes
from matplotlib.patches import Rectangle
from torch.utils.data.dataloader import DataLoader
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.transforms import Compose, ToPILImage, ToTensor

CAPTURED_LABELS = [
    'ego vehicle',
    'vehicle',
    'pedestrian',
    'license plate',
    'traffic light',
    'traffic sign',
    'pole',
    'polegroup',
    'fence',
    'guard rail'
]

COLOR_DICT = []

LABEL_DICT = {
    'car': 'vehicle',
    'bus': 'vehicle',
    'truck': 'vehicle',
    'caravan': 'vehicle',
    'trailer': 'vehicle',
    'motorcycle': 'vehicle',
    'train': 'vehicle',
    'bicycle': 'vehicle',
    'person': 'pedestrian',
    'rider': 'pedestrian',
}

# ALL_LABELS = ['ground', 'road', 'sidewalk', 'parking',
#               'rail track', 'building', 'wall', 'bridge', 'tunnel']


def show_image(image, target=None, mode: Literal['polygon', 'bb', None] = None):
    if type(image) == torch.Tensor:
        transform = ToPILImage()
        image = transform(image)

    plt.figure()
    plt.imshow(np.array(image, dtype=np.int32))

    if target != None:
        objects = target["objects"]
        for object in objects:
            polygon = object['polygon']

            if object["label"] == 'cargroup':
                object_name = 'car'
            else:
                object_name = object["label"]

            object_class = [
                c for c in cityscpapes.Cityscapes.classes if c.name == object_name][0]

            r, g, b = object_class.color
            color = tuple([r/255, g/255, b/255])

            x, y = zip(*polygon)
            if mode == 'polygon':
                plt.plot(x, y, color=color)
            elif mode == 'bb':
                x_min = np.min(x)
                x_max = np.max(x)
                y_min = np.min(y)
                y_max = np.max(y)

                plt.gca().add_patch(Rectangle((x_min, y_min), x_max - x_min,
                                              y_max - y_min, facecolor='none', lw=2, edgecolor=color))
                plt.gca().text(x_min, y_max, object_name, fontsize=8, color='white',
                               bbox={'facecolor': color, 'edgecolor': 'none'})

    plt.show()


def show_output(image, real_data, pred_data):
    transform = ToPILImage()
    image = transform(image)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(np.array(image, dtype=np.int32))
    ax2.imshow(np.array(image, dtype=np.int32))

    pred_boxes = pred_data[0]['boxes'].data.tolist()
    pred_labels = pred_data[0]['labels'].data.tolist()

    for box, label in zip(pred_boxes, pred_labels):
        random.seed(label)
        color = tuple([
            random.randint(0, 255)/255,
            random.randint(0, 255)/255,
            random.randint(0, 255)/255
        ])
        str_label = CAPTURED_LABELS[label]

        ax1.title.set_text("Predictions")

        ax1.add_patch(Rectangle((box[0], box[1]), box[2] - box[0],
                                box[3] - box[1], facecolor='none', lw=2, edgecolor=color))
        ax1.text(box[0], box[3], str_label, fontsize=8, color='white',
                 bbox={'facecolor': color, 'edgecolor': 'none'})

    real_boxes = real_data[0]['boxes'].data.tolist()
    real_labels = real_data[0]['labels'].data.tolist()

    for box, label in zip(real_boxes, real_labels):
        random.seed(label)
        color = tuple([
            random.randint(0, 255)/255,
            random.randint(0, 255)/255,
            random.randint(0, 255)/255
        ])
        str_label = CAPTURED_LABELS[label]

        ax2.title.set_text("Ground Truth")

        ax2.add_patch(Rectangle((box[0], box[1]), box[2] - box[0],
                                box[3] - box[1], facecolor='none', lw=2, edgecolor=color))
        ax2.text(box[0], box[3], str_label, fontsize=8, color='white',
                 bbox={'facecolor': color, 'edgecolor': 'none'})

    plt.show()


def train(epochs: int, num_images: int, model: MaskRCNN, train_loader):
    for epoch in range(epochs):
        print(f'Epoch: {epoch}')
        losses = []
        for i, data in enumerate(train_loader, 0):
            if (i == num_images):
                print('Finished Images')
                break

            print(f'Batch: {i}')

            images, targets = data

            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            losses.append(loss)

    print(losses)
    return losses


def save_model(model):
    print('Saving model...')
    BASE_PATH = 'models/'
    ver = 1
    invalid_path = True

    while invalid_path:
        filename = f'model_v1.{ver}'
        path = BASE_PATH + filename

        if not os.path.isfile(path):
            invalid_path = False

        ver = ver + 1

    torch.save(model, path)
    print('Model saved!')


def collate_fn(data):
    images = []
    targets = []

    for datapoint in data:
        images.append(datapoint[0].to(device))
        targets.append(datapoint[1])

    return images, targets


def transform_target(data):
    s = time.time()
    print('Transforming target')
    image = np.array(data[0])
    objects = data[1]['objects']

    boxes = []
    labels = []

    for object in objects:
        # get object label
        str_label: str = object['label']
        if str_label.endswith('group'):
            str_label = str_label.replace('group', '')

        if str_label in LABEL_DICT:
            str_label = LABEL_DICT[str_label]

        if str_label in CAPTURED_LABELS:
            label = CAPTURED_LABELS.index(str_label)
            # get object bounding box
            polygon = object['polygon']
            x, y = zip(*polygon)
            box = [min(x), min(y), max(x), max(y)]
            boxes.append(box)
            labels.append(label)

    masks = []
    for label_id in np.unique(labels):
        mask = np.array(image == label_id, dtype=np.uint8)
        masks.append(mask)

    target = {}
    target['masks'] = torch.as_tensor(masks, dtype=torch.uint8).to(device)
    target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32).to(device)
    target['labels'] = torch.as_tensor(labels, dtype=torch.int64).to(device)

    e = time.time()
    print(f"{e-s}s")
    return target


if __name__ == "__main__":
    print('Getting device...')
    device = torch.device(
        0) if torch.cuda.is_available() else torch.device('cpu')
    print(f'Running on {device}')

    print('Loading data...')
    train_data = cityscpapes.Cityscapes(
        'datasets/Cityscape', 'train', 'fine', ['semantic', 'polygon'], target_transform=transform_target, transform=Compose([ToTensor()]))

    val_data = cityscpapes.Cityscapes(
        'datasets/Cityscape', 'val', 'fine', ['semantic', 'polygon'], target_transform=transform_target, transform=Compose([ToTensor()]))

    train_loader = DataLoader(
        train_data, batch_size=1, num_workers=0, collate_fn=collate_fn)

    val_loader = DataLoader(
        val_data, batch_size=1, num_workers=0, collate_fn=collate_fn)

    # print('Data classes')
    NUM_CLASSES = len(CAPTURED_LABELS)
    # print([cityScapesClass.name for cityScapesClass in cityscpapes.Cityscapes.classes])

    print('Loading model...')
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        num_classes=NUM_CLASSES).to(device)

    print('Training model...')
    train(1, 4, model, train_loader)

    print('Evaluating model...')
    model.eval()

    t_data = next(iter(val_loader))
    images, targets = t_data

    out = model(images)

    show_output(images[0], targets, out)

    save_model(model)
