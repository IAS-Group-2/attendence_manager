import cv2
import os

def extract_frames(video_path, frames_dir, overwrite=False, start=-1, end=-1, every=1):
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    if start == -1:
        start = 0
    if end == -1:
        end = os.path.getsize(video_path)
    count = 0
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if count % every == 0:
                cv2.imwrite(os.path.join(frames_dir, str(count) + '.jpg'), frame)
                count += 1
        else:
            break
    cap.release()
    return count

extract_frames("input/vid1.mp4", "input/frames")

