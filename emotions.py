import torch
import cv2
import numpy as np
import traceback

import torchvision.transforms.functional as TF

from timeit import default_timer

from detector import Detector
from utils.torchbp import Classifier
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from train_emotion_model import create_model


class Recognizer:
    def __init__(self, face_size=128, show_frame=False):
        self.show_frame = show_frame

        self.detector = Detector()

        # parameters for loading data and images
        self.emotion_model_path = './models/emotion_model_new.pth'
        self.emotion_labels = {0: 'Neutral', 1: 'Happiness', 2: 'Sadness', 3: 'Surprise', 4: 'Fear', 5: 'Disgust',
                      6: 'Anger', 7: 'Contempt', 8: 'None', 9: 'Uncertain', 10: 'No-face'}
        self.labelstr = ','.join([self.emotion_labels[i] for i in range(8)])

        # loading models
        self.emotion_classifier = Classifier(self.load_model(), len(self.emotion_labels))
        self.face_size = face_size

    def load_model(self):
        model = create_model()
        model.load_state_dict(torch.load(self.emotion_model_path))
        return model


    def prepare_face_image(self, face_image):
        face_image = TF.to_tensor(face_image)
        face_image = TF.normalize(face_image, [0.5] * 3, [0.5] * 3, inplace=True)
        return face_image.unsqueeze(0)

    def recognize_video(self, path=None, csv=True):
        # starting video streaming

        if self.show_frame:
            cv2.namedWindow('window_frame')

        # Select video or webcam feed
        cap = None
        if path is None:
            cap = cv2.VideoCapture(0)  # Webcam source
        elif isinstance(path, str):
            cap = cv2.VideoCapture(path)  # Video file source
        else:
            raise Exception('path argument must be a string')

        emotions_by_frames = [] # container for recording emotion probabilities by frame
        frame_counter = 0
        while cap.isOpened(): # True:
            try:

                ret, bgr_image = cap.read()
                if not ret:
                    break
                #print('Frame {} is read'.format(frame_counter))
                frame_counter += 1

                rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

                start = default_timer()
                faces = self.detector.detect_faces(rgb_image)
                #print('faces', faces)
                end = default_timer()
                #print('face detection:', end - start)

                if len(faces) == 0:
                    emotions_by_frames.append(np.array([0] * 7))

                for face_coordinates in faces:
                    # face_coordinates: [x, y, w, h]
                    face_coordinates = list(map(int, face_coordinates))
                    x, y, w, h = face_coordinates

                    # expand to 20%
                    padding_size_x, padding_size_y = int(0.2*x), int(0.2*y)
                    w += padding_size_x
                    h += padding_size_y
                    x -= padding_size_x // 2
                    y -= padding_size_y // 2

                    face_image = rgb_image[y:y+h, x:x+w]
                    try:
                        face_image = cv2.resize(face_image, (self.face_size, self.face_size))
                    except:
                        emotions_by_frames.append(np.array([0] * len(self.emotion_labels)))
                        continue

                    face_image = self.prepare_face_image(face_image)
                    #print('Shape:', face_image.shape)

                    emotion_prediction = self.emotion_classifier.predict(face_image, probs=True)[0].numpy()
                    #print('Prediction:', emotion_prediction)
                    emotion_probability = np.max(emotion_prediction)
                    if len(emotion_prediction) > 0:
                        emotions_by_frames.append(emotion_prediction)
                    else:
                        emotions_by_frames.append(np.array([0]*len(self.emotion_labels)))

                    emotion_label_arg = np.argmax(emotion_prediction)
                    emotion_text = self.emotion_labels[emotion_label_arg]

                    if emotion_text == 'angry':
                        color = emotion_probability * np.asarray((255, 0, 0))
                    elif emotion_text == 'sad':
                        color = emotion_probability * np.asarray((0, 0, 255))
                    elif emotion_text == 'happy':
                        color = emotion_probability * np.asarray((255, 255, 0))
                    elif emotion_text == 'surprise':
                        color = emotion_probability * np.asarray((0, 255, 255))
                    else:
                        color = emotion_probability * np.asarray((0, 255, 0))

                    color = color.astype(int)
                    color = color.tolist()

                    draw_bounding_box(face_coordinates, bgr_image, color)
                    draw_text(face_coordinates, bgr_image, emotion_text, color, 0, -45, 1, 1)

                #bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                cv2.imshow('window_frame', bgr_image)
                if (cv2.waitKey(1) & 0xFF == ord('q')) and self.show_frame:
                    break
            except:
                traceback.print_exc()
                break

        cap.release()
        cv2.destroyAllWindows()
        if csv:
            return self.to_csv(emotions_by_frames)
        return emotions_by_frames

    def to_csv(self, emotion_probs):
        for i in range(len(emotion_probs)):
            emotion_probs[i] = ','.join(map(str, [i + 1] + emotion_probs[i].tolist()))

        return '\n'.join([self.labelstr, *emotion_probs])


if __name__ == '__main__':
    """cap = cv2.VideoCapture(0)

    detector = Detector()

    while True:
        ret, bgr_image = cap.read()
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb_image)

        for face_coordinates in faces:
            face_coordinates = list(map(int, face_coordinates))
            print(face_coordinates)"""

    recognizer = Recognizer()
    recognizer.recognize_video()

    import argparse

    parser = argparse.ArgumentParser(description='Video emotion recognizer.')
    parser.add_argument('--path', type=str, help='path directory with video to process', default='D:/Projects/Emotion_recognition/webcam-base64-streaming-master/results/04ec3e93')
    args = parser.parse_args()
    recognizer = Recognizer()
    res = recognizer.recognize_video(args.path + '/face_record.mp4')#'./demo/test_video_short.mp4')
    with open(args.path + '/result.csv', 'w') as f:
        print(res, file=f)