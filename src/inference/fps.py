import time

import cv2


# Visualization parameters
row_size = 20  # pixels
left_margin = 24  # pixels
text_color = (0, 0, 255)  # red
font_size = 1
font_thickness = 1


def calculate_fps(start_time, fps_avg_frame_count):
    fps = fps_avg_frame_count / (time.time() - start_time)
    return fps


def draw_fps(image, fps):
    # Show the FPS
    fps_text = "FPS = " + str(int(fps))
    text_location = (left_margin, row_size)
    return cv2.putText(
        image,
        fps_text,
        text_location,
        cv2.FONT_HERSHEY_PLAIN,
        font_size,
        text_color,
        font_thickness,
    )
