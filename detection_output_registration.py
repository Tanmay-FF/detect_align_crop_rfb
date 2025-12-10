import numpy as np
from onnxruntime_extensions import onnx_op, PyCustomOpDef



#------------------------------------ Detection Output Registration ------------------------------------

@onnx_op(op_type="DetectionOutput", inputs=[PyCustomOpDef.dt_float, PyCustomOpDef.dt_float, PyCustomOpDef.dt_float], 
         outputs=[PyCustomOpDef.dt_float])
def detection_output_operator(boxes, confidences, priorboxes, num_classes=2, confidence_threshold=0.01, nms_threshold=0.15, top_k=100, keep_top_k=50):
    """
    Custom DetectionOutput operator for SSD.
    Args:
        boxes: (1, 39936) array of box offsets.
        confidences: (1, 19968) array of confidence scores.
        priorboxes: (1, 2, 54408) array of prior box coordinates.
        num_classes: Number of classes in the detection task.
        confidence_threshold: Minimum confidence score for a detection to be valid.
        nms_threshold: IoU threshold for Non-Maximum Suppression.
        top_k: Number of top-scoring boxes to keep before NMS.
        keep_top_k: Final number of boxes to keep after NMS.
    Returns:
        detections: Final filtered detections with shape (keep_top_k, 7),
                    where each row is [batch_id, class_id, score, xmin, ymin, xmax, ymax].
    """
    
    def decode_boxes(boxes, priorboxes):
        print("BOX SHAPE: ", boxes.shape)
        # Decode the box predictions: priorboxes shape is [N, 2, total_boxes]
        prior_boxes = priorboxes[0, 0, :].reshape(-1, 4)  # Get first part (center_x, center_y, width, height)
        prior_variances = priorboxes[0, 1, :].reshape(-1, 4)  # Get second part (variances)
        
        centers = prior_boxes[:, :2] + boxes[:, :2] * prior_boxes[:, 2:] * prior_variances[:, :2]
        sizes = np.exp(boxes[:, 2:] * prior_variances[:, 2:]) * prior_boxes[:, 2:]
        return np.concatenate([centers - sizes / 2, centers + sizes / 2], axis=1)

    def nms(boxes, scores, threshold):
        # Perform Non-Maximum Suppression
        indices = np.argsort(-scores)
        keep = []
        while len(indices) > 0:
            i = indices[0]
            keep.append(i)
            ious = compute_iou(boxes[i], boxes[indices[1:]])
            indices = indices[1:][ious <= threshold]
        return keep

    def compute_iou(box, other_boxes):
        # Compute IoU between a box and multiple boxes
        inter_xmin = np.maximum(box[0], other_boxes[:, 0])
        inter_ymin = np.maximum(box[1], other_boxes[:, 1])
        inter_xmax = np.minimum(box[2], other_boxes[:, 2])
        inter_ymax = np.minimum(box[3], other_boxes[:, 3])
        inter_area = np.maximum(0, inter_xmax - inter_xmin) * np.maximum(0, inter_ymax - inter_ymin)
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        other_areas = (other_boxes[:, 2] - other_boxes[:, 0]) * (other_boxes[:, 3] - other_boxes[:, 1])
        union_area = box_area + other_areas - inter_area
        return inter_area / np.maximum(union_area, 1e-6)

    # Reshape `boxes` and `confidences` to match the expected formats
    boxes = boxes.reshape(-1, 4)  # Reshape to [num_boxes, 4]
    confidences = confidences.reshape(-1, num_classes)  # Reshape to [num_boxes, num_classes]
    print(boxes.shape)
    print(boxes[0])
    # Decode boxes using priorboxes
    decoded_boxes = decode_boxes(boxes, priorboxes)

    # Initialize outputs
    all_detections = []

    for class_id in range(1, num_classes):  # Skip background class (class_id=0)
        class_scores = confidences[:, class_id]
        mask = class_scores > confidence_threshold
        filtered_boxes = decoded_boxes[mask]
        filtered_scores = class_scores[mask]

        if len(filtered_scores) == 0:
            continue

        # Perform NMS
        selected_indices = nms(filtered_boxes, filtered_scores, nms_threshold)
        selected_boxes = filtered_boxes[selected_indices]
        selected_scores = filtered_scores[selected_indices]

        # Append detections
        for i in range(len(selected_boxes)):
            all_detections.append([0, class_id, selected_scores[i], *selected_boxes[i]])

    # Sort and keep top detections
    all_detections = np.array(all_detections)
    if len(all_detections) > keep_top_k:
        top_indices = np.argsort(-all_detections[:, 2])[:keep_top_k]
        all_detections = all_detections[top_indices]

    # Ensure the output shape is (keep_top_k, 7) and type float32
    output_detections = all_detections.astype(np.float32)
    
    return output_detections