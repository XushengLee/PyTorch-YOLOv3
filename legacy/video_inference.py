from __future__ import division
import cv2
from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
import torch.nn.functional as F
import os
import sys
import time
import datetime
import argparse
import tqdm
from utils.datasets import pad_to_square
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
from vis import vis_frame
from utils.utils import rescale_boxes

def infer_video(model, video_path, save_path, iou_thres, conf_thres, nms_thres, img_size, batch_size=1):
    model.eval()
    if os.path.isfile(video_path):
        cap = cv2.VideoCapture(video_path)
    else:
        raise Exception('no such video')
    tp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    stream = cv2.VideoWriter(os.path.join(save_path, 'out.mp4'), cv2.VideoWriter_fourcc(*'MJPG'), 20.0, (int(cap.get(3)), int(cap.get(4))) )
    transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                           ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(frame.shape)
            img = transform(frame).to(device)
            print(img.shape)
            img, _ = pad_to_square(img,0)
            print(img.shape)
            img = F.interpolate(img.unsqueeze(0), img_size, mode='nearest').squeeze()
            print(img.shape)
            img = img.unsqueeze(0)
            with torch.no_grad():
                output = model(img)
                output = non_max_suppression(output, conf_thres=conf_thres, nms_thres=nms_thres)
                output = rescale_boxes(output[0], img_size, frame.shape[:2])
#                print(output)
               # print(len(output[0][0]))
                img = vis_frame(frame, output)                   
                stream.write(img)
        else:
            break
    cap.release() 
    stream.release()
            
        

def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()
        
    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--video", type=str, default='examples/PETS09-S2L2.mp4')
    parser.add_argument('--save', type=str, default='examples/res/')
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.data_config)
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    infer_video(model, opt.video, opt.save, iou_thres=opt.iou_thres, conf_thres=opt.conf_thres, nms_thres=opt.nms_thres, img_size=opt.img_size, batch_size=1)
#    print("Compute mAP...")
#
#    precision, recall, AP, f1, ap_class = evaluate(
#        model,
#        path=valid_path,
#        iou_thres=opt.iou_thres,
#        conf_thres=opt.conf_thres,
#        nms_thres=opt.nms_thres,
#        img_size=opt.img_size,
#        batch_size=8,
#    )

#    print("Average Precisions:")
#    for i, c in enumerate(ap_class):
#        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

#    print(f"mAP: {AP.mean()}")
