import cv2
import matplotlib.pyplot as plt

#----------------- bounding box--------------------------------------------

def plot_bounding_boxes(image_path, bboxes, labels=None, colors=None, save_path= None):
    """
    Plots bounding boxes on an image.

    Args:
        image_path (str): Path to the image.
        bboxes (list of tuples): List of bounding boxes [(x_min, y_min, x_max, y_max), ...].
        labels (list of str, optional): List of labels corresponding to each bounding box. Default is None.
        colors (list of tuples, optional): List of RGB colors for the bounding boxes. Default is None.
    """
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB for displaying
    h, w, _ = image.shape  # Get image dimensions
    print(h,w)
    
    if colors is None:
        colors = [(255, 0, 0)] * len(bboxes)  # Default color (red) if not provided

    # Draw bounding boxes
    for i, (bbox, color) in enumerate(zip(bboxes, colors)):
        # If coordinates are normalized, convert to absolute pixel values
        x_min, y_min, x_max, y_max = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        x_min_pixel = int(x_min * w) if x_min < 1  else int(x_min)
        y_min_pixel = int(y_min * h) if y_min < 1  else int(y_min)
        x_max_pixel = int(x_max * w) if x_max <= 1 else int(x_max)
        y_max_pixel = int(y_max * h) if y_max <= 1 else int(y_max)

        # Ensure coordinates are within valid image dimensions
        x_min_pixel = max(0, min(x_min_pixel, w))
        y_min_pixel = max(0, min(y_min_pixel, h))
        x_max_pixel = max(0, min(x_max_pixel, w))
        y_max_pixel = max(0, min(y_max_pixel, h))
        
        print(x_min_pixel, y_min_pixel, x_max_pixel, y_max_pixel)

        cv2.rectangle(image, (x_min_pixel, y_min_pixel), (x_max_pixel, y_max_pixel), color, 2)

        if labels and i < len(labels):
            cv2.putText(
                image,
                labels[i],
                (x_min_pixel, y_min_pixel - 10),  # Text position (above the box)
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                color,
                2,
                cv2.LINE_AA,
            )

    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    
    # Save the image if save_path is provided
    if save_path:
        save_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back to BGR for saving
        cv2.imwrite(save_path, save_image)
        print(f"Image saved to {save_path}")