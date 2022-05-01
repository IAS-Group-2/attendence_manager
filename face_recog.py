import face_recognition
import cv2

class FaceRecognition:
    def __init__(self, train_images) -> None:
        """ train_images = [("sarthak", image0), ("patidar", image1), ...] """
        self.embeddings = {}
        self.add_embeddings(train_images)
    
    def get_embedding(self, image):
        """ return embedding of the image """
        return face_recognition.face_encodings(image)[0]

    def add_embeddings(self, images):
        """ input images = [("sarthak", image0), ("patidar", image1), ...] """
        for name, image in images:
            self.embeddings[name] = self.get_embedding(image)
    
    def recognise(self, frames):
        """ input = list of frames
        return set of names = {"sarthak", "unknown", "negi", "gagan", ...} in no particular order """
        known_names, known_embeddings = list(self.embeddings.keys()), list(self.embeddings.values()) 
        pred_names = set()
        for frame in frames:   
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            # print(type(known_embeddings))

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_embeddings, face_encoding)
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_names[first_match_index]
                    pred_names.add(name)
        
        return set(pred_names)