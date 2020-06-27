from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
import io
import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import cv2
from utils.datasets import pad_to_square
from utils.utils import rescale_boxes
import os
import matplotlib as mpl
#mpl.rcParams['savefig.pad_inches'] = 0
plt.switch_backend('agg')

def fig2img(fig):
    buf = io.BytesIO()
    fig.tight_layout()
    plt.autoscale(tight=True)
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    parser.add_argument("--video_path", type=str, default="examples/PETS09-S2L2.mp4")
    parser.add_argument("--save_path", type=str, default="examples/res/")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mode
    
    classes = load_classes(opt.class_path)  # Extracts class labels from file


    cap = cv2.VideoCapture(opt.video_path)
    stream = cv2.VideoWriter(os.path.join(opt.save_path, 'out.mp4'), cv2.VideoWriter_fourcc(*'MJPG'),
                             20.0, (int(cap.get(3)), int(cap.get(4))) )
    
    trans = transforms.Compose([
                                    transforms.ToTensor(),
#                                    transforms.Normalize(mean=[0.486, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ])
    plt.figure()
    plt.margins(0, 0)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            img = trans(frame)

            img, _ = pad_to_square(img, 0)
            img = F.interpolate(img.unsqueeze(0), opt.img_size, mode='bilinear').squeeze()
            img = img.unsqueeze(0).to(device)
            with torch.no_grad():
                detections = model(img)
                detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
                # Bounding-box colors
                cmap = plt.get_cmap("tab20b")
                colors = [cmap(i) for i in np.linspace(0, 1, 20)]
                # Create plot
#                img = np.array(frame)
#                plt.figure()
                fig, ax = plt.subplots(1)
                ax.imshow(np.array(frame))

                # Draw bounding boxes and labels of detections
                if detections[0] is not None:
                    # Rescale boxes to original image
                    detections = rescale_boxes(detections[0], opt.img_size, frame.size[::-1])
                    unique_labels = detections[:, -1].cpu().unique()
                    n_cls_preds = len(unique_labels)
                    bbox_colors = random.sample(colors, n_cls_preds)
                    print('-----------------')
                    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                        print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

                        box_w = x2 - x1
                        box_h = y2 - y1

                        color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                        # Create a Rectangle patch
                        bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                        # Add the bbox to the plot
                        ax.add_patch(bbox)
                        # Add label
                        plt.text(
                            x1,
                            y1,
                            s=classes[int(cls_pred)],
                            color="white",
                            verticalalignment="top",
                            bbox={"color": color, "pad": 0},
                        )

                # Save generated image with detections
                plt.axis("off")
                plt.gca().xaxis.set_major_locator(NullLocator())
                plt.gca().yaxis.set_major_locator(NullLocator())
#                plt.savefig(f"output/1.png", bbox_inches="tight", pad_inches=0.0)
                plt.tight_layout(pad=0)
                plt.autoscale(tight=True)
                fig.canvas.draw()
                im = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                im = im.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
#                im = fig2img(fig)
#                im = im.resize((int(cap.get(3)), int(cap.get(4))))
                print(frame.size)
#                print('xx', im.shape)
                print(fig.canvas.get_width_height())
#                im = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
                im = cv2.resize(im, (int(cap.get(3)), int(cap.get(4))), interpolation=cv2.INTER_LINEAR)
                print(im.shape)
                stream.write(im)
                plt.close(fig)           
        else:
            break
    cap.release()
    stream.release()



