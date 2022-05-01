"""

"""
import face_recognition
import cv2
from face_recog import FaceRecognition


if __name__ == "__main__":
    image_1 = face_recognition.load_image_file("attendance/test/g2.png")
    image_2 = face_recognition.load_image_file("attendance/test/k1.png")
    image_3 = face_recognition.load_image_file("attendance/test/s2.png")

    images = [("gagan", image_1), ("karan", image_2), ("sarthak", image_3)]

    fr_class_1 = FaceRecognition(images)

    # print(fr_class_1.embeddings["sarthak"])
    # print(fr_class_1.embeddings["karan"])
    # print(fr_class_1.embeddings["gagan"])

    frames = []
    for i in range(0,100):
        frames.append(cv2.imread("attendance/input/frames/"+str(i)+".jpg"))
    
    print(fr_class_1.recognise(frames))