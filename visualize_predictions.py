import os
import numpy as np
import cv2
import argparse
import onnxruntime as ort
from onnxruntime_extensions import get_library_path
from torchvision import transforms
from PIL import Image
import prior_box_registration as pb
import detection_output_registration as do
# ------------------------------------------------------------------------------
# ARGUMENTS & CONFIG
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser('CPU Face Detection & Visualization')
parser.add_argument('--image_folder', type=str, default=r'D:\Face-Recognition\face_recognition_data_yogesh\test_data_to_be_used\wakefern_raw_masked\white_female\1281', help='Path to images')
parser.add_argument('--detector', type=str, default='onnx_models/model_with_output_v5.onnx', help='Detector ONNX')
parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold')
args = parser.parse_args()

# Setup ONNX Session (Optimized for your CPU)
so = ort.SessionOptions()
so.register_custom_ops_library(get_library_path())
so.intra_op_num_threads = 0 # Use all available cores on your i9
session = ort.InferenceSession(args.detector, sess_options=so, providers=['CPUExecutionProvider'])

transform = transforms.Compose([
    transforms.Resize((360, 480)),
    transforms.ToTensor() 
])

# ------------------------------------------------------------------------------
# CORE FUNCTIONS
# ------------------------------------------------------------------------------
def detect_and_visualize(image_path):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None: return
    
    orig_h, orig_w = img_bgr.shape[:2]
    
    # 1. Preprocess
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    transformed = transform(img_pil)
    image_array = transformed.unsqueeze(0).numpy().astype(np.float32)

    # 2. Run Inference
    input_name = session.get_inputs()[0].name
    outputs = session.run(["detection_out"], {input_name: image_array})
    
    # Standard 'detection_out' shape is [1, 1, N, 7]
    # Format per detection: [batch_id, class_id, score, x1, y1, x2, y2]
    detections = np.squeeze(outputs[0]) 

    # 3. Filter and Draw
    for i in range(detections.shape[0]):
        detection = detections[i]
        score = detection[2]
        
        if score > args.threshold:
            # Denormalize coordinates to original image pixels
            x1 = int(detection[3] * orig_w)
            y1 = int(detection[4] * orig_h)
            x2 = int(detection[5] * orig_w)
            y2 = int(detection[6] * orig_h)

            # Draw Box (Green)
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw Confidence Score
            label = f"Face: {score:.2f}"
            cv2.putText(img_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 4. Display or Save
    cv2.imshow('Face Detection Results', img_bgr)
    # Wait for key; 'q' to quit, any other key for next image
    if cv2.waitKey(0) & 0xFF == ord('q'):
        return False
    return True

# ------------------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    if not os.path.exists(args.image_folder):
        print(f"Error: Folder {args.image_folder} not found.")
    else:
        for root, _, files in os.walk(args.image_folder):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    path = os.path.join(root, file)
                    print(f"Processing: {file}")
                    if not detect_and_visualize(path):
                        break
        cv2.destroyAllWindows()