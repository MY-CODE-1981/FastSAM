#!/home/initial/.pyenv/versions/3.8.6/envs/fpose/bin/python
# -*- coding: utf-8 -*-
import cv2
import message_filters
import numpy as np
import os
import rospy
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
import sys
import time
import yaml

import argparse
from fastsam import FastSAM, FastSAMPrompt 
import ast
import torch
# from PIL import Image
from utils.tools import convert_box_xywh_to_xyxy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="./weights/FastSAM.pt", help="model"
    )
    parser.add_argument(
        "--img_path", type=str, default="./images/dogs.jpg", help="path to image file"
    )
    parser.add_argument("--imgsz", type=int, default=640, help="image size")
    parser.add_argument(
        "--iou",
        type=float,
        default=0.9,
        help="iou threshold for filtering the annotations",
    )
    parser.add_argument(
        "--text_prompt", type=str, default=None, help='use text prompt eg: "a dog"'
    )
    parser.add_argument(
        "--conf", type=float, default=0.4, help="object confidence threshold"
    )
    parser.add_argument(
        "--output", type=str, default="./output/", help="image save path"
    )
    parser.add_argument(
        "--randomcolor", type=bool, default=True, help="mask random color"
    )
    parser.add_argument(
        "--point_prompt", type=str, default="[[0,0]]", help="[[x1,y1],[x2,y2]]"
    )
    parser.add_argument(
        "--point_label",
        type=str,
        default="[0]",
        help="[1,0] 0:background, 1:foreground",
    )
    parser.add_argument("--box_prompt", type=str, default="[[0,0,0,0]]", help="[[x,y,w,h],[x2,y2,w2,h2]] support multiple boxes")
    parser.add_argument(
        "--better_quality",
        type=str,
        default=False,
        help="better quality using morphologyEx",
    )
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    parser.add_argument(
        "--device", type=str, default=device, help="cuda:[0,1,2,3,4] or cpu"
    )
    parser.add_argument(
        "--retina",
        type=bool,
        default=True,
        help="draw high-resolution segmentation masks",
    )
    parser.add_argument(
        "--withContours", type=bool, default=False, help="draw the edges of the masks"
    )
    return parser.parse_args()


import time
 
class ObjectDetecton:
    def __init__(self, args):
        # load model
        self.model = FastSAM(args.model_path)
        args.point_prompt = ast.literal_eval(args.point_prompt)
        args.box_prompt = convert_box_xywh_to_xyxy(ast.literal_eval(args.box_prompt))
        args.point_label = ast.literal_eval(args.point_label)
        
        from cv_bridge import CvBridge
        self.bridge = CvBridge()

        sub_image_raw = message_filters.Subscriber("/camera/color/image_raw", Image)
        sub_camera_info = message_filters.Subscriber("/camera/color/camera_info", CameraInfo)
        ts = message_filters.ApproximateTimeSynchronizer([sub_image_raw, sub_camera_info], 2, 0.05)
        ts.registerCallback(self.callback)
        
    def callback(self, image_raw, camera_info):
        print("callback")
        
        img_cv2 = self.bridge.imgmsg_to_cv2(image_raw, desired_encoding='passthrough')
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        
        ##########################
        print("")
        start_time = time.time()
        # input = Image.open(args.img_path)
        input = img_cv2 # input.convert("RGB")
        everything_results = self.model(
            img_cv2,
            device=args.device,
            retina_masks=args.retina,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou    
            )
        bboxes = None
        points = None
        point_label = None
        prompt_process = FastSAMPrompt(input, everything_results, device=args.device)
        if args.box_prompt[0][2] != 0 and args.box_prompt[0][3] != 0:
            print("0-1")
            ann = prompt_process.box_prompt(bboxes=args.box_prompt)
            bboxes = args.box_prompt
        elif args.text_prompt != None:
            print("0-2")
            ann = prompt_process.text_prompt(text=args.text_prompt)
        elif args.point_prompt[0] != [0, 0]:
            print("0-3")
            ann = prompt_process.point_prompt(
                points=args.point_prompt, pointlabel=args.point_label
            )
            points = args.point_prompt
            point_label = args.point_label
        else:
            print("0-4")
            ann = prompt_process.everything_prompt()
            
        filename = time.strftime("%Y%m%d_%H%M%S")
        
        # prompt_process.plot(
        #     annotations=ann,
        #     output_path=args.output+str(filename)+".jpg",
        #     bboxes = bboxes,
        #     points = points,
        #     point_label = point_label,
        #     withContours=args.withContours,
        #     better_quality=args.better_quality,
        # )
    
        result = prompt_process.plot_only(
            annotations=ann,
            output_path=args.output+str(filename)+".jpg",
            bboxes = bboxes,
            points = points,
            point_label = point_label,
            withContours=args.withContours,
            better_quality=args.better_quality,
        )
        
        cv2.imshow("1", result)
        cv2.waitKey(1)

if __name__ == '__main__':
    args = parse_args()
    rospy.init_node("inference_d435i")
    objDet = ObjectDetecton(args)
    rospy.spin()
