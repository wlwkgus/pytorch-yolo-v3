from __future__ import division
import time
from collections import OrderedDict

from torch.autograd import Variable

from sun_dataset import get_data_loader
from util import *
import argparse
import os
import os.path as osp
from darknet import Darknet
from preprocess import prep_image, inp_to_image
import visdom
import cv2
import time
import math


def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

    parser.add_argument("--images", dest='images', help=
    "Image / Directory containing images to perform detection upon",
                        default="imgs", type=str)
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1, type=int)
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--score_thresh", default=0.7)
    parser.add_argument(
        "--num_preprocess_workers", dest="num_preprocess_workers", help="num preprocess workers", default=4)
    parser.add_argument("--cfg", dest='cfgfile', help=
    "Config file",
                        default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help=
    "weightsfile",
                        default="yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=str)
    parser.add_argument("--lr", dest='lr', help="learning rate", default=1e-4, type=float)
    parser.add_argument("--weight_decay", dest='weight_decay', help='weight decay', default=1e-5, type=float)
    parser.add_argument("--ckpt_dir", dest='ckpt_dir', help="ckpt dir", default="ckpt", type=str)
    parser.add_argument("--display_port", dest='display_port', help="display port", default=8001, type=int)

    parser.add_argument("--scales", dest="scales", help="Scales to use for detection",
                        default="1,2,3", type=str)
    parser.add_argument("--is_train", dest="is_train", help="is train",
                        default="1", type=int)
    parser.add_argument("--load_epoch", dest="is_train", help="is train",
                        default="170", type=int)
    parser.add_argument("--soft_nms", type=int, default=1, type=int)

    return parser.parse_args()


class PlotManager(object):
    def __init__(self):
        self.vis = visdom.Visdom(port=args.display_port)
        self.name = 'Object detection loss'
        self.display_id = 0
        self.image_titles = []

    def plot_errors(self, epoch, counter_ratio, errors):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(errors.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([errors[k] for k in self.plot_data['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
            Y=np.array(self.plot_data['Y']),
            opts={
                'title': self.name + ' loss over time',
                'legend': self.plot_data['legend'],
                'xlabel': 'step',
                'ylabel': 'loss'},
            win=self.display_id)

    def plot_image(self, image_numpy, title):
        if title not in self.image_titles:
            self.image_titles.append(title)
        self.vis.image(image_numpy, opts=dict(title=title), win=self.display_id + 1 + self.image_titles.index(title))


def relu(value):
    if value > 0:
        return value
    else:
        return 0.

def get_iou(box1, box2):
    iou_x1 = max(
        box1[0] - box1[2] / 2,
        box2[0] - box2[2] / 2
    )
    iou_y1 = max(
        box1[1] - box1[3] / 2,
        box2[1] - box2[3] / 2
    )
    iou_x2 = min(
        box1[0] + box1[2] / 2,
        box2[0] + box2[2] / 2
    )
    iou_y2 = min(
        box1[1] + box1[3] / 2,
        box2[1] + box2[3] / 2
    )
    return relu(iou_x2 - iou_x1) * relu(iou_y2 - iou_y1)

if __name__ == '__main__':
    # Visdom setting here
    args = arg_parse()
    plot_manager = PlotManager()

    scales = args.scales
    scales = [int(x) for x in scales.split(',')]

    scales_indices = []
    args.reso = int(args.reso)
    num_boxes = [args.reso // 8, args.reso // 16, args.reso // 32]
    num_boxes = sum([3 * (x ** 2) for x in num_boxes])

    for scale in scales:
        li = list(range((scale - 1) * num_boxes // 3, scale * num_boxes // 3))
        scales_indices.extend(li)

    images = args.images
    batch_size = int(args.bs)
    confidence = float(args.confidence)
    CUDA = torch.cuda.is_available()

    num_classes = 1
    # classes = load_classes('data/coco.names')

    # Set up the neural network
    print("Loading network.....")
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    # TODO : Load from checkpoint here.
    save_filename = 'yolo_net' + '-' + str(args.load_epoch)
    save_path = os.path.join(args.ckpt_dir, save_filename)
    model.load_state_dict(torch.load(save_path))
    if CUDA:
        model.cuda(device='cuda:0')


    print("Network successfully loaded")

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    # If there's a GPU availible, put the model on GPU
    if CUDA:
        model.cuda(device='cuda:0')

    optm = torch.optim.Adam(
        params=model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    data_loader = get_data_loader(args)

    for epoch in range(180):
        for step, batch in enumerate(data_loader):
            # load the image
            start = time.time()
            if CUDA:
                batch['img'] = batch['img'].cuda()
                batch['label'] = batch['label'].cuda()

            # Apply offsets to the result predictions
            # Tranform the predictions as described in the YOLO paper
            # flatten the prediction vector
            # B x (bbox cord x no. of anchors) x grid_w x grid_h --> B x bbox x (all the boxes)
            # Put every proposed box as a row.
            prediction = model(Variable(batch['img'], requires_grad=True), CUDA)
            prediction.requires_grad = True

            prediction = prediction[:, scales_indices]
            single_prediction = prediction[0]

            # prediction : batch_size x boxes x 85
            # single_prediction : boxes x 85

            # TODO : NMS here.
            indexes = torch.topk(single_prediction[..., 4], k=prediction.size(0), dim=0)[1]
            sorted_single_prediction = single_prediction[indexes]
            in_out_flags = torch.ones(sorted_single_prediction.size(0))
            for i in range(sorted_single_prediction.size(0)):
                if in_out_flags[i] < 1:
                    continue
                if args.soft_nms:
                    for j in range(i+1, sorted_single_prediction.size(0)):
                        iou = get_iou(
                            sorted_single_prediction[i].numpy(),
                            sorted_single_prediction[j].numpy()
                        )
                        sorted_single_prediction[j][4] *= math.exp(-iou * iou / 0.5)
                    in_out_flags = sorted_single_prediction[4] > args.score_thresh
                else:
                    for j in range(i+1, sorted_single_prediction.size(0)):
                        iou = get_iou(
                            sorted_single_prediction[i].numpy(),
                            sorted_single_prediction[j].numpy()
                        )
                        if iou > args.nms_thresh:
                            in_out_flags[j] = 0
                    in_out_flags *= (sorted_single_prediction[4] > args.score_thresh)

            filtered_prediction = sorted_single_prediction[in_out_flags.nonzero()]

            color = (80, 7, 65)
            single_label = batch['label'][0]
            single_image_cv_format = (batch['img'][0].cpu().numpy() * 255).astype(np.uint8).transpose(1, 2,
                                                                                                      0).copy()
            batch['img'].cuda()
            single_image_cv_format = single_image_cv_format[..., ::-1]
            prediction_image = single_image_cv_format.copy()
            ground_truth_image = single_image_cv_format.copy()

            # TODO : save these images

            for j in range(filtered_prediction.size(0)):
                x_center, y_center, w, h = tuple(filtered_prediction[j][0:4].int())
                prediction_image = cv2.rectangle(
                    prediction_image.copy(),
                    (int(x_center - w / 2), int(y_center - h / 2)),
                    (int(x_center + w / 2), int(y_center + h / 2)),
                    color,
                    1
                )
            plot_manager.plot_image(np.transpose(prediction_image[..., ::-1], (2, 0, 1)), 'prediction')
            del prediction_image
            for j in range(single_label.size(0)):
                if single_label[j][4].int() == 0:
                    continue
                x_min, y_min, x_max, y_max = tuple(single_label[j][0:4].int())
                ground_truth_image = cv2.rectangle(
                    ground_truth_image.copy(),
                    (x_min, y_min),
                    (x_max, y_max),
                    color,
                    1
                )
            plot_manager.plot_image(np.transpose(ground_truth_image[..., ::-1], (2, 0, 1)), 'ground truth')
            del ground_truth_image

            time.sleep(5)
