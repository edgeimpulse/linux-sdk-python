import numpy as np
try:
    import cv2
except ImportError:
    print('Missing OpenCV, install via `pip3 install "opencv-python>=4.5.1.48,<5"`')
    exit(1)
from edge_impulse_linux.image import get_features_from_image_with_studio_mode


def create_test_image(frame_buffer_cols, frame_buffer_rows):
    # Create an empty image with 3 channels (RGB)
    image_rgb888_packed = np.zeros((frame_buffer_rows, frame_buffer_cols, 3), dtype=np.uint8)

    for row in range(frame_buffer_rows):
        for col in range(frame_buffer_cols):
            # Change color a bit (light -> dark from top->down, so we know if the image looks good quickly)
            blue_intensity = int((255.0 / frame_buffer_rows) * row)
            green_intensity = int((255.0 / frame_buffer_cols) * col)

            # Set the pixel values
            image_rgb888_packed[row, col, 0] = blue_intensity  # Blue channel
            image_rgb888_packed[row, col, 1] = green_intensity  # Green channel
            image_rgb888_packed[row, col, 2] = 0  # Red channel is zero for test

    return image_rgb888_packed

# %%

def demo_mode(mode):
    frame_buffer_rows = 480
    frame_buffer_cols = 640
    test_image = create_test_image(frame_buffer_cols, frame_buffer_rows)

    _, test_image = get_features_from_image_with_studio_mode(test_image, mode, 200,200, False)

    # Display the image using OpenCV and ensure it stays open
    cv2.imshow(mode, test_image)
    cv2.waitKey(0)  # Wait for a key press to close the image window
    cv2.destroyAllWindows()  # Close the image window
    cv2.waitKey(1)  # Patch for macOS

demo_mode('fit-shortest')
demo_mode('fit-longest')
demo_mode('squash')
