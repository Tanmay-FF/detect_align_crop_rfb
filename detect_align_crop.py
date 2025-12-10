#import required libraries

import os
import torch
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from time import time
from torchvision import transforms

import onnxruntime as ort
from onnxruntime_extensions import get_library_path
import bounding_box as bb
import prior_box_registration as pb
import detection_output_registration as do
import argparse

parser= argparse.ArgumentParser('parser for face detector inference')

parser.add_argument('--image_folder', type= str, default='test_data',
                    required= False,  help= 'provide image path for face detection')

parser.add_argument('--detector', type= str, default= 'onnx_models/model_with_output_v5.onnx',
                    required= False, help= 'provide onnx face detector path')

parser.add_argument('--landmark', type= str, default= 'onnx_models/landmark_model.onnx',
                    required= False, help= 'provide onnx landmark detector path')

parser.add_argument('--output_folder', type= str, default= 'crop_after_detection_alignment',
                    required= False, help= 'provide onnx landmark detector path')


args= parser.parse_args()

# Load and execute the ONNX model with the custom operator
so = ort.SessionOptions()
so.register_custom_ops_library(get_library_path())  # Register the custom operator library

# Create an inference session for detection model
session = ort.InferenceSession(args.detector, sess_options=so, providers=["CUDAExecutionProvider"])

# Create an inference session for landmark model
session_lm    = ort.InferenceSession(args.landmark,sess_options=so)

transform = transforms.Compose([
            transforms.Resize((360,480)),
            transforms.ToTensor()  # range [0, 255] -> [0.0,1.0]
            # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])


def postprocess(out):
    # Convert list to numpy array if it's not already
    out = np.array(out)
    # print(out.shape)
    box  = out[:,3:7]
    cls  = out[:,1]
    conf = out[:,2]
    return (box, conf, cls)

def filter_bbox(bboxes, threshold= 0.5):
    # threshold= 0.9
    filtered_bboxes= []
    
    for bbox in bboxes:
        conf= bbox[2]
        if conf > threshold:
            filtered_bboxes.append(bbox)
            
    return filtered_bboxes

def detect(image_array):
    inputs     = {session.get_inputs()[0].name:image_array}
    start= time()     
    output = session.run(["detection_out"], inputs)
    # print(output)
    # print(output[0].shape)
    end= time()
    filtered_output= filter_bbox(output[0]) 
    if(len(filtered_output)==0):
            return None, None, None
    box, conf, cls = postprocess(filtered_output)
    return box, conf, end-start


TEMPLATE = np.float32([
    (0.0792396913815, 0.339223741112), (0.0829219487236, 0.456955367943),
    (0.0967927109165, 0.575648016728), (0.122141515615, 0.691921601066),
    (0.168687863544, 0.800341263616), (0.239789390707, 0.895732504778),
    (0.325662452515, 0.977068762493), (0.422318282013, 1.04329000149),
    (0.531777802068, 1.06080371126), (0.641296298053, 1.03981924107),
    (0.738105872266, 0.972268833998), (0.824444363295, 0.889624082279),
    (0.894792677532, 0.792494155836), (0.939395486253, 0.681546643421),
    (0.96111933829, 0.562238253072), (0.970579841181, 0.441758925744),
    (0.971193274221, 0.322118743967), (0.163846223133, 0.249151738053),
    (0.21780354657, 0.204255863861), (0.291299351124, 0.192367318323),
    (0.367460241458, 0.203582210627), (0.4392945113, 0.233135599851),
    (0.586445962425, 0.228141644834), (0.660152671635, 0.195923841854),
    (0.737466449096, 0.182360984545), (0.813236546239, 0.192828009114),
    (0.8707571886, 0.235293377042), (0.51534533827, 0.31863546193),
    (0.516221448289, 0.396200446263), (0.517118861835, 0.473797687758),
    (0.51816430343, 0.553157797772), (0.433701156035, 0.604054457668),
    (0.475501237769, 0.62076344024), (0.520712933176, 0.634268222208),
    (0.565874114041, 0.618796581487), (0.607054002672, 0.60157671656),
    (0.252418718401, 0.331052263829), (0.298663015648, 0.302646354002),
    (0.355749724218, 0.303020650651), (0.403718978315, 0.33867711083),
    (0.352507175597, 0.349987615384), (0.296791759886, 0.350478978225),
    (0.631326076346, 0.334136672344), (0.679073381078, 0.29645404267),
    (0.73597236153, 0.294721285802), (0.782865376271, 0.321305281656),
    (0.740312274764, 0.341849376713), (0.68499850091, 0.343734332172),
    (0.353167761422, 0.746189164237), (0.414587777921, 0.719053835073),
    (0.477677654595, 0.706835892494), (0.522732900812, 0.717092275768),
    (0.569832064287, 0.705414478982), (0.635195811927, 0.71565572516),
    (0.69951672331, 0.739419187253), (0.639447159575, 0.805236879972),
    (0.576410514055, 0.835436670169), (0.525398405766, 0.841706377792),
    (0.47641545769, 0.837505914975), (0.41379548902, 0.810045601727),
    (0.380084785646, 0.749979603086), (0.477955996282, 0.74513234612),
    (0.523389793327, 0.748924302636), (0.571057789237, 0.74332894691),
    (0.672409137852, 0.744177032192), (0.572539621444, 0.776609286626),
    (0.5240106503, 0.783370783245), (0.477561227414, 0.778476346951)])


TPL_MIN, TPL_MAX = np.min(TEMPLATE, axis=0), np.max(TEMPLATE, axis=0)
MINMAX_TEMPLATE = (TEMPLATE - TPL_MIN) / (TPL_MAX - TPL_MIN)
INNER_EYES_AND_BOTTOM_LIP = [39, 42, 57]
# print(MINMAX_TEMPLATE(INNER_EYES_AND_BOTTOM_LIP))


def align(imgDim1,imgDim2,rgbImg, landmarks):
        assert rgbImg is not None
        assert landmarks is not None
        
        npLandmarks = np.float32(landmarks)    
        landmarkIndices = INNER_EYES_AND_BOTTOM_LIP
        npLandmarkIndices = np.array(landmarkIndices)

        T=MINMAX_TEMPLATE[npLandmarkIndices]
        T[:,0]=imgDim1*T[:,0]
        T[:,1]=imgDim2*T[:,1]
        H = cv2.getAffineTransform(npLandmarks, imgDim1 * MINMAX_TEMPLATE[npLandmarkIndices])
        thumbnail = cv2.warpAffine(rgbImg, H, (imgDim1, imgDim2))
        
        # Transform landmarks to the aligned image space
        transformed_landmarks = cv2.transform(np.array([npLandmarks]), H)[0]

        return thumbnail, transformed_landmarks


def detect_landmark(ori_img_path, bbox_output_folder='crop_before_alignment'):
    os.makedirs(bbox_output_folder, exist_ok=True)
    filename = os.path.basename(ori_img_path).split('.')[0]
    img_bgr = cv2.imread(ori_img_path)
    if img_bgr is None:
        return None, None, None, None, None
    image_h, image_w, _ = img_bgr.shape
    # print(f'height & width: {image_h}, {image_w}')
    
    # Convert BGR to RGB (since OpenCV loads images in BGR format)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # Convert the NumPy array to a PIL Image
    img_pil = Image.fromarray(img_rgb)
    # Apply transformations (assuming `transform` is a torchvision transform pipeline)
    transformed_image = transform(img_pil)
    # Convert transformed image to NumPy array and add batch dimension
    image_array = np.expand_dims(np.array(transformed_image), axis=0)
    # Convert to float32
    image_array = image_array.astype(np.float32)
    
          
    boxes,score, time_taken    = detect(image_array)

    
    if boxes is None:
        return None, None, None, None, None
    
    # c = 0
    # for i in range(boxes):
    #      if c < 5:
    #         print(i)
    #         c = c + 1

    lm= []
    tt_landmark= 0
    # print(f"boxes shape: {boxes.shape}")
    # # print(type(score), score)
    # labels = [f"face_onnx:{score[0]:.3f}"]  # Example labels with confidence score
    # colors = [(255, 0, 0)]  # Colors: red and green
    # bb.plot_bounding_boxes(ori_img_path, boxes, labels, colors)
    
    for i, bbox in enumerate(boxes):
        # If coordinates are normalized, convert to absolute pixel values
        x_min, y_min, x_max, y_max = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        x_min_pixel = int(x_min * image_w) if x_min < 1  else int(x_min)
        y_min_pixel = int(y_min * image_h) if y_min < 1  else int(y_min)
        x_max_pixel = int(x_max * image_w) if x_max <= 1 else int(x_max)
        y_max_pixel = int(y_max * image_h) if y_max <= 1 else int(y_max)

        # Ensure coordinates are within valid image dimensions
        xmin = max(0, min(x_min_pixel, image_w))
        ymin = max(0, min(y_min_pixel, image_h))
        xmax = max(0, min(x_max_pixel, image_w))
        ymax = max(0, min(y_max_pixel, image_h))
        print(xmin, ymin, xmax, ymax)
        h, w   = ymax - ymin , xmax - xmin     #height & width of cropped image
        print(h,w)
        scaler = np.array([h, w])
        
        if xmin >= xmax or ymin >= ymax:
            print(f"Invalid box skipped: {bbox}")
            continue

        crop = img_rgb[ymin:ymax, xmin:xmax]
        if crop.size == 0:
            print("Empty crop skipped.")
            continue
            
        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        # Save the image with landmarks
        bbox_output_path = os.path.join(bbox_output_folder, filename) + '_' + str(i) + '.jpg'
        print(f"Saving bbox cropped image at {bbox_output_folder}")
        cv2.imwrite(bbox_output_path, crop_bgr)
        
        if crop.size == 0:
            continue  # Skip empty crops

        crop_resized = cv2.resize(crop, (64, 64))
        crop_resized = np.expand_dims(crop_resized, axis=0)  # Add batch dim
        # Ensure the resized image is in uint8 format
        crop_resized = crop_resized.astype(np.uint8)
        
        inputs     = {session_lm.get_inputs()[0].name:crop_resized}
        start= time()
        ort_outputs= session_lm.run(None, inputs)
        end= time()
        tt_landmark += (end-start)
        keypoints  = np.array(ort_outputs).reshape(98,2)
        landmarks  = (keypoints * scaler) + (ymin, xmin)
        
        
        landmarks_xy = []
        lm_cnt=0
        for y , x in landmarks:
                lm_cnt += 1
                if(lm_cnt==65 or lm_cnt==69 or lm_cnt==86):
                    landmarks_xy.append([x , y])
        lm.append(landmarks_xy)

    return boxes, score, lm, time_taken, tt_landmark


# output_folder will contain images after detection, cropping & alignment

def detect_align_crop(input_folder, output_folder='crop_after_detection_alignment'):
    
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)
    #     print(f"Folder created: {output_folder}")
    # else:
    #     print(f"Folder already exists: {output_folder}")

    avg_inf= 0
    avg_inf_landmark= 0
    count= 0
    
    for directory, folder, files in os.walk(input_folder):
        for file in files:
            image_path = os.path.join(directory, file)
            image_bgr = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            
            bboxs, scores, landmarks, time_taken, time_taken_landmark = detect_landmark(image_path, bbox_output_folder=output_folder) 
            if time_taken is not None:
                count += 1
                avg_inf += time_taken
                avg_inf_landmark += time_taken_landmark

            if bboxs is None:
                continue


            for i in range(len(landmarks)):
                filename   = os.path.basename(image_path).split('.')[0]
                output_path= os.path.join(output_folder, filename) + '_' + str(i) + '.jpg'
                aligned_img, transformed_landmarks = align(112, 112, image_rgb, landmarks[i])
                cv2.imwrite(output_path, cv2.cvtColor(aligned_img, cv2.COLOR_RGB2BGR))

    # print(f'average inference time taken for detection: {avg_inf/count}')
    # print(f'average inference time taken for landmark detection: {avg_inf_landmark/count}')


detect_align_crop(args.image_folder, args.output_folder)


