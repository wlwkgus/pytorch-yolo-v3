from __future__ import division
import time
from torch.autograd import Variable

from sun_dataset import get_data_loader
from util import *
import argparse
import os
import os.path as osp
from darknet import Darknet
from preprocess import prep_image, inp_to_image


def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

    parser.add_argument("--images", dest='images', help=
    "Image / Directory containing images to perform detection upon",
                        default="imgs", type=str)
    parser.add_argument("--bs", dest="bs", help="Batch size", default=2)
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
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

    parser.add_argument("--scales", dest="scales", help="Scales to use for detection",
                        default="1,2,3", type=str)
    parser.add_argument("--is_train", dest="is_train", help="is train",
                        default="1", type=int)

    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()

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

    num_classes = 5841
    # classes = load_classes('data/coco.names')

    # Set up the neural network
    print("Loading network.....")
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    print("Network successfully loaded")

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    # If there's a GPU availible, put the model on GPU
    if CUDA:
        model.cuda()

    optm = torch.optim.Adam(
        params=model.parameters(),
        lr=1e-4,
        weight_decay=1e-5,
    )

    data_loader = get_data_loader(args)

    for epoch in range(100):
        for i, batch in enumerate(data_loader):
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
            prediction = model(Variable(batch['img']), CUDA)

            prediction = prediction[:, scales_indices]

            # prediction : batch_size x boxes x 85
            # ground truth : batch_size x boxes x 85
            # label : batch_size x variant boxes x 85

            # batch size x boxes x variant boxes x 85
            expanded_prediction = prediction.unsqueeze(2)
            expanded_prediction = expanded_prediction.repeat(1, 1, batch['label'].size(1), 1)
            expanded_batch_label = batch['label'].unsqueeze(1).repeat(1, prediction.size(1), 1, 1)

            iou_x1 = torch.max(
                torch.cat(
                    [
                        (expanded_prediction[..., 0] - expanded_prediction[..., 2] / 2).unsqueeze(3),
                        expanded_batch_label[..., 0].unsqueeze(3)
                    ], dim=3)

            )
            iou_y1 = torch.max(
                torch.cat(
                    [
                        (expanded_prediction[..., 1] - expanded_prediction[..., 3] / 2).unsqueeze(3),
                        expanded_batch_label[..., 1].unsqueeze(3)
                    ], dim=3)

            )
            iou_x2 = torch.min(
                torch.cat(
                    [
                        (expanded_prediction[..., 0] + expanded_prediction[..., 2] / 2).unsqueeze(3),
                        expanded_batch_label[..., 2].unsqueeze(3)
                    ], dim=3)

            )
            iou_y2 = torch.min(
                torch.cat(
                    [
                        (expanded_prediction[..., 1] + expanded_prediction[..., 3] / 2).unsqueeze(3),
                        expanded_batch_label[..., 3].unsqueeze(3)
                    ], dim=3)
            )
            # intersection tensor : batch_size x boxes x variant boxes
            intersection_tensor = torch.abs(iou_x1 - iou_x2) * torch.abs(iou_y1 - iou_y2)
            union_tensor = (expanded_prediction[..., 2] * expanded_prediction[..., 3]) +\
                           (
                               torch.abs(batch['label'][..., 0] - batch['label'][..., 2]) *
                               torch.abs(batch['label'][..., 1] - batch['label'][..., 3])
                           ) - intersection_tensor

            # iou_tensor : batch_size x boxes x variant boxes
            iou_tensor = intersection_tensor / union_tensor

            # top_iou_value : batch_size x boxes x 1
            top_iou_value, top_iou_index = torch.topk(iou_tensor, 1, dim=2)

            # label : batch_size x variant boxes x 85
            # is_greater_than_iou_threshold : batch_size x boxes
            is_greater_than_iou_threshold = Variable((top_iou_value.squeeze() > 0.5).float(), requires_grad=False)
            # ground truth : batch_size x boxes x 85
            ground_truth = Variable(
                torch.gather(
                    batch['label'],
                    1,
                    top_iou_index.repeat(1, 1, batch['label'].size(2)).long()
                ),
                requires_grad=False
            )

            # if iou is lower than threshold, set objectness score zero.
            ground_truth[..., 4] *= is_greater_than_iou_threshold

            # all the losses here.
            # coordinate loss
            coordinate_loss = torch.sum(
                (prediction[..., :4] - ground_truth[..., :4]) * (prediction[..., :4] - ground_truth[..., :4])
            ) / (prediction.size(0) * prediction.size(1))
            # objectness loss
            objectness_loss = torch.nn.BCELoss()(prediction[..., 4], ground_truth[..., 4])
            # class loss
            class_loss = torch.nn.BCELoss()(
                prediction[..., 5:] * is_greater_than_iou_threshold,
                ground_truth[..., 5:] * is_greater_than_iou_threshold
            )

            total_loss = coordinate_loss + objectness_loss + class_loss

            optm.zero_grad()
            total_loss.backward()
            optm.step()
            torch.cuda.empty_cache()
            end = time.time()
            print("[epoch : {}] : step {} loss = {} / {} elapsed".format(epoch, i, total_loss, end-start))
