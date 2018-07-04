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


def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

    parser.add_argument("--images", dest='images', help=
    "Image / Directory containing images to perform detection upon",
                        default="imgs", type=str)
    parser.add_argument("--bs", dest="bs", help="Batch size", default=2, type=int)
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
    parser.add_argument("--lr", dest='lr', help="learning rate", default=1e-4, type=float)
    parser.add_argument("--weight_decay", dest='weight_decay', help='weight decay', default=1e-5, type=float)
    parser.add_argument("--ckpt_dir", dest='ckpt_dir', help="ckpt dir", default="ckpt", type=str)
    parser.add_argument("--display_port", dest='display_port', help="display port", default=8001, type=int)

    parser.add_argument("--scales", dest="scales", help="Scales to use for detection",
                        default="1,2,3", type=str)
    parser.add_argument("--is_train", dest="is_train", help="is train",
                        default="1", type=int)

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

            # prediction : batch_size x boxes x 85
            # ground truth : batch_size x boxes x 85
            # label : batch_size x variant boxes x 85

            # batch size x boxes x variant boxes x 85
            expanded_prediction = prediction.unsqueeze(2).cuda(device='cuda:0')
            expanded_prediction = expanded_prediction.repeat(1, 1, batch['label'].size(1), 1)
            expanded_batch_label = batch['label'].unsqueeze(1).repeat(1, prediction.size(1), 1, 1)

            iou_x1 = torch.max(
                torch.cat(
                    [
                        (expanded_prediction[..., 0] - expanded_prediction[..., 2] / 2).unsqueeze(3),
                        expanded_batch_label[..., 0].unsqueeze(3)
                    ], dim=3)
                , dim=3
            )[0]
            iou_y1 = torch.max(
                torch.cat(
                    [
                        (expanded_prediction[..., 1] - expanded_prediction[..., 3] / 2).unsqueeze(3),
                        expanded_batch_label[..., 1].unsqueeze(3)
                    ], dim=3)
                , dim=3
            )[0]
            iou_x2 = torch.min(
                torch.cat(
                    [
                        (expanded_prediction[..., 0] + expanded_prediction[..., 2] / 2).unsqueeze(3),
                        expanded_batch_label[..., 2].unsqueeze(3)
                    ], dim=3)
                , dim=3
            )[0]
            iou_y2 = torch.min(
                torch.cat(
                    [
                        (expanded_prediction[..., 1] + expanded_prediction[..., 3] / 2).unsqueeze(3),
                        expanded_batch_label[..., 3].unsqueeze(3)
                    ], dim=3)
                , dim=3
            )[0]
            # intersection tensor : batch_size x boxes x variant boxes
            intersection_tensor = torch.nn.ReLU()(iou_x2 - iou_x1) * torch.nn.ReLU()(iou_y2 - iou_y1)
            union_tensor = (expanded_prediction[..., 2] * expanded_prediction[..., 3]) +\
                           (
                               torch.abs(expanded_batch_label[..., 0] - expanded_batch_label[..., 2]) *
                               torch.abs(expanded_batch_label[..., 1] - expanded_batch_label[..., 3])
                           ) - intersection_tensor

            # iou_tensor : batch_size x boxes x variant boxes
            iou_tensor = intersection_tensor / union_tensor
            iou_tensor *= expanded_batch_label[..., 4]
            # transposed iou tensor : batch_size x variant boxes x boxes
            # transposed_iou_tensor = torch.transpose(iou_tensor, 1, 2).contiguous()

            # TODO : assign top iou for each ground truth box.
            # top_iou_value : batch_size x boxes x 1
            top_iou_value, top_iou_index = torch.topk(iou_tensor, 1, dim=2)
            # transposed_top_iou_value : batch_size x variant_boxes x 1
            transposed_top_iou_value, transposed_top_iou_index = torch.topk(iou_tensor, 1, dim=1)
            transposed_top_iou_value = torch.transpose(transposed_top_iou_value, 1, 2).contiguous()
            transposed_top_iou_index = torch.transpose(transposed_top_iou_index, 1, 2).contiguous()

            # label : batch_size x variant boxes x 85
            # ground_truth_is_greater_than_iou_threshold : batch_size x boxes
            # is_greater_than_iou_threshold : batch_size x variant boxes
            is_greater_than_iou_threshold = Variable((transposed_top_iou_value.squeeze() > 0.5).float(), requires_grad=False)
            ground_truth_is_greater_than_iou_threshold = Variable((top_iou_value.squeeze() > 0.5).float(), requires_grad=False)
            # ground truth : batch_size x boxes x 85
            ground_truth = Variable(
                torch.gather(
                    batch['label'],
                    1,
                    top_iou_index.repeat(1, 1, batch['label'].size(2)).long().cuda(device='cuda:0')
                ),
                requires_grad=False
            )
            selected_prediction = Variable(
                torch.gather(
                    prediction,
                    1,
                    transposed_top_iou_index.repeat(1, 1, prediction.size(2)).long().cuda(device="cuda:0")
                )
            )

            # if iou is lower than threshold, set objectness score zero.
            # ground_truth[..., 4].data = torch.squeeze(top_iou_value).data
            # ground_truth[..., 4] *= is_greater_than_iou_threshold

            # all the losses here.
            # coordinate loss
            coordinate_loss = torch.sum(
                (
                    selected_prediction[..., 0] - Variable((batch['label'][..., 0] + batch['label'][..., 2]) / 2)
                ) * (
                    selected_prediction[..., 0] - Variable((batch['label'][..., 0] + batch['label'][..., 2]) / 2)
                ) * is_greater_than_iou_threshold * Variable(batch['label'][..., 4])
                + (
                    selected_prediction[..., 1] - Variable((batch['label'][..., 1] + batch['label'][..., 3]) / 2)
                ) * (
                    selected_prediction[..., 1] - Variable((batch['label'][..., 1] + batch['label'][..., 3]) / 2)
                ) * is_greater_than_iou_threshold * Variable(batch['label'][..., 4])
                + (
                    selected_prediction[..., 2] - Variable(batch['label'][..., 2] - batch['label'][..., 0])
                ) * (
                    selected_prediction[..., 2] - Variable(batch['label'][..., 2] - batch['label'][..., 0])
                ) * is_greater_than_iou_threshold * Variable(batch['label'][..., 4])
                + (
                    selected_prediction[..., 3] - Variable(batch['label'][..., 3] - batch['label'][..., 1])
                ) * (
                    selected_prediction[..., 3] - Variable(batch['label'][..., 3] - batch['label'][..., 1])
                ) * is_greater_than_iou_threshold * Variable(batch['label'][..., 4])
            ) / (selected_prediction.size(0) * selected_prediction.size(1))

            # coordinate_loss = torch.sum(
            #     (
            #         prediction[..., 0] - (ground_truth[..., 0] + ground_truth[..., 2]) / 2
            #     ) * (
            #         prediction[..., 0] - (ground_truth[..., 0] + ground_truth[..., 2]) / 2
            #     ) * ground_truth_is_greater_than_iou_threshold
            #     + (
            #         prediction[..., 1] - (ground_truth[..., 1] + ground_truth[..., 3]) / 2
            #     ) * (
            #         prediction[..., 1] - (ground_truth[..., 1] + ground_truth[..., 3]) / 2
            #     ) * ground_truth_is_greater_than_iou_threshold
            #     + (
            #         prediction[..., 2] - (ground_truth[..., 2] - ground_truth[..., 0])
            #     ) * (
            #         prediction[..., 2] - (ground_truth[..., 2] - ground_truth[..., 0])
            #     ) * ground_truth_is_greater_than_iou_threshold
            #     + (
            #         prediction[..., 3] - (ground_truth[..., 3] - ground_truth[..., 1])
            #     ) * (
            #         prediction[..., 3] - (ground_truth[..., 3] - ground_truth[..., 1])
            #     ) * ground_truth_is_greater_than_iou_threshold
            # ) / (prediction.size(0) * prediction.size(1))
            # coordinate_loss.requires_grad = True
            # objectness loss
            # objectness_loss = torch.nn.BCELoss()(
            #     prediction[..., 4] * ground_truth[..., 4],
            #     ground_truth[..., 4] * ground_truth_is_greater_than_iou_threshold
            # )
            objectness_loss = torch.nn.BCELoss()(
                selected_prediction[..., 4] * Variable(batch['label'][..., 4]),
                Variable(batch['label'][..., 4]) * is_greater_than_iou_threshold
            )
            # class loss
            if num_classes == 1:
                total_loss = coordinate_loss + objectness_loss
            else:
                # class_loss = torch.nn.BCELoss()(
                #     prediction[..., 5:] * is_greater_than_iou_threshold,
                #     ground_truth[..., 5:] * is_greater_than_iou_threshold
                # )
                class_loss = Variable(torch.FloatTensor([0.]))
                total_loss = coordinate_loss + objectness_loss + class_loss

            optm.zero_grad()
            total_loss.backward()
            optm.step()
            torch.cuda.empty_cache()
            end = time.time()
            if step % 100 == 0:
                print("[epoch : {}] : step {} loss = {} / {} elapsed".format(epoch, step, total_loss, end-start))
                plot_manager.plot_errors(epoch * len(data_loader) + step, 0, OrderedDict([
                    ('batch_loss', total_loss.cpu().data.numpy()),
                    ('none', 0.)
                ]))
                total_loss.cuda(device='cuda:0')
                color = (80, 7, 65)
                single_prediction = prediction[0]
                single_ground_truth = ground_truth[0]
                single_image_cv_format = (batch['img'][0].cpu().numpy() * 255).astype(np.uint8).transpose(1, 2,
                                                                                                          0).copy()
                batch['img'].cuda()
                single_image_cv_format = single_image_cv_format[..., ::-1]
                prediction_image = single_image_cv_format.copy()
                ground_truth_image = single_image_cv_format.copy()
                for j in range(single_prediction.size(0)):
                    if single_prediction[j][4] < 0.5:
                        continue
                    x_center, y_center, w, h = tuple(single_prediction[j][0:4].int())
                    prediction_image = cv2.rectangle(
                        prediction_image.copy(),
                        (int(x_center - w / 2), int(y_center - h / 2)),
                        (int(x_center + w / 2), int(y_center + h / 2)),
                        color,
                        1
                    )
                plot_manager.plot_image(np.transpose(prediction_image[..., ::-1], (2, 0, 1)), 'prediction')
                del prediction_image
                for j in range(single_prediction.size(0)):
                    x_min, y_min, x_max, y_max = tuple(single_ground_truth[j][0:4].int())
                    ground_truth_image = cv2.rectangle(
                        ground_truth_image.copy(),
                        (x_min, y_min),
                        (x_max, y_max),
                        color,
                        1
                    )
                plot_manager.plot_image(np.transpose(ground_truth_image[..., ::-1], (2, 0, 1)), 'ground truth')
                del ground_truth_image

        if epoch % 10 == 0:
            save_filename = 'yolo_net' + '-' + str(epoch)
            save_path = os.path.join(args.ckpt_dir, save_filename)
            torch.save(model.cpu().state_dict(), save_path)
            if CUDA:
                model.cuda(device='cuda:0')
