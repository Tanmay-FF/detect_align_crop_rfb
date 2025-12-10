import numpy as np
from onnxruntime_extensions import onnx_op


#-----------------------priorbox registration-----------------------------------------------

@onnx_op(op_type="PriorBox", attrs=[
             "min_sizes",     #: PyCustomOpDef.dt_float,     # List of floats
             "max_sizes",     #: PyCustomOpDef.dt_float,     # List of floats
             "aspect_ratios", #: PyCustomOpDef.dt_float,     # List of floats
             "variances",     #: PyCustomOpDef.dt_float
             "clip",          #: PyCustomOpDef.dt_int64,     # Integer
             "flip",          #: PyCustomOpDef.dt_int64,     # Integer
             "steps",         #: PyCustomOpDef.dt_float,     # List of floats
             "offset"         #: PyCustomOpDef.dt_float      # Single float
])
def prior_box(feature_map: np.ndarray, **kwargs) -> np.ndarray:
    """
    Custom PriorBox operator that generates anchor boxes.

    Args:
        feature_map (np.ndarray): Shape of the feature map (N, C, H, W).
        **kwargs: Dictionary of PriorBox attributes.

    Returns:
        np.ndarray: Generated prior boxes (anchors) as a 2D array.
    """

    # Extract attributes from kwargs
    min_sizes    = (kwargs['min_sizes'])
    max_sizes    = (kwargs['max_sizes'])
    variances    = (kwargs['variances'])
    steps        = (kwargs['steps'])
    aspect_ratios= (kwargs['aspect_ratios'])
    clip         = float(kwargs['clip'])
    flip         = float(kwargs['flip'])
    offset       = float(kwargs['offset'])
    
    # Split the string and convert to float
    min_sizes     = [float(num) for num in min_sizes.split()]
    max_sizes     = [float(num) for num in max_sizes.split()]
    variances     = [float(num) for num in variances.split()]
    steps         = [float(num) for num in steps.split()]
    aspect_ratios = [float(num) for num in aspect_ratios.split()]
    
    
    # Convert min_sizes and max_sizes to list of floats
    if isinstance(min_sizes, (list, tuple)):
        min_sizes = [float(size) for size in min_sizes]
    else:
        raise TypeError("min_sizes must be a list or tuple of numbers")

    if isinstance(max_sizes, (list, tuple)):
        max_sizes = [float(size) for size in max_sizes]
    else:
        raise TypeError("max_sizes must be a list or tuple of numbers")

    fmap_height, fmap_width = feature_map.shape[2], feature_map.shape[3]
    img_height, img_width = kwargs.get("img_height", 360), kwargs.get("img_width", 480)
    
    prior_boxes = []

    # Generate boxes for each cell in the feature map
    for i in range(fmap_height):
        for j in range(fmap_width):
            center_x = ((j + offset) * steps[0])/ img_width
            center_y = ((i + offset) * steps[1])/ img_height

            # Generate boxes for each min_size and corresponding max_size
            for k, min_size in enumerate(min_sizes):
                box_width = min_size / img_width
                box_height = min_size / img_height
                prior_boxes.append([center_x, center_y, box_width, box_height])

                # Add box with max_size if it exists
                if k < len(max_sizes):
                    max_size = max_sizes[k]
                    box_width = max_size / img_width
                    box_height = max_size / img_height
                    prior_boxes.append([center_x, center_y, box_width, box_height])


                # Add boxes for aspect ratios
                for ar in aspect_ratios:
                    if ar == 1.0:
                        continue
                    box_width = min_size * np.sqrt(ar) / img_width
                    box_height = min_size / np.sqrt(ar) / img_height
                    prior_boxes.append([center_x, center_y, box_width, box_height])

    # Convert prior_boxes to numpy array
    prior_boxes = np.array(prior_boxes, dtype=np.float32)

    # Clip boxes to the range [0, 1] if clip is enabled
    if clip:
        prior_boxes = np.clip(prior_boxes, 0, 1)

    # Generate variances as the same shape as coordinates
    num_boxes = len(prior_boxes)
    variances_array = np.tile(variances, (num_boxes, 1))
    
    # Combine coordinates and variances into separate channels
    coordinates_channel = prior_boxes  # Shape: (num_boxes, 4)
    variances_channel = variances_array  # Shape: (num_boxes, 4)

    # Stack channels along axis 0
    combined = np.stack((coordinates_channel, variances_channel), axis=0)  # Shape: (2, num_boxes, 4)

    # Reshape to (1, 2, num_boxes * 4)
    reshaped = combined.reshape(1, 2, -1)  # Shape: (1, 2, num_boxes * 4)

    return reshaped