# import the necessary packages
import cv2
ds_factor = 0.6
import cv2
import numpy as np
import tensorflow.compat.v1 as tf # import tensorflow as tf
import datetime
import argparse
import keras

MODEL_NAME = 'hand_inference_graph'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
CLASSES = ['garbage', 'next', 'start', 'stop']
PATH_TO_CNN_MODEL = "model/hand_detector_15_Epoch.h5" # model we have 


def detect_objects(image_np, detection_graph, sess):
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name(
        'detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name(
        'detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name(
        'detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name(
        'num_detections:0')

    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores,
            detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    return np.squeeze(boxes), np.squeeze(scores)


def draw_box_on_image(num_hands_detect, score_thresh, scores, boxes, im_width, im_height, image_np):
    for i in range(num_hands_detect):
        if (scores[i] > score_thresh):
            (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                          boxes[i][0] * im_height, boxes[i][2] * im_height)
            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))
            cv2.rectangle(image_np, p1, p2, (77, 255, 9), 3, 1)


def load_inference_graph():
    # load frozen tensorflow model into memory
    print("***************************Loading hand frozen graph***************************")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)
    print("***************************Loaded Hand model graph***************************")
    return detection_graph, sess

def load_KerasGraph(path): 
    print("***************************Loading Model***************************")
    thread_graph = tf.Graph()
    with thread_graph.as_default():
        thread_session = tf.Session()
        with thread_session.as_default():
            model = keras.models.load_model(path)
            #model._make_predict_function()
            graph = tf.get_default_graph()
    print("***************************Model loaded***************************")
    return model, graph, thread_session

def get_box_image(num_hands_detect, score_thresh, scores, boxes, im_width, im_height, image_np):
    for i in range(num_hands_detect):
        if (scores[i] > score_thresh):
            (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                          boxes[i][0] * im_height, boxes[i][2] * im_height)
            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))
            return image_np[int(top):int(bottom), int(left):int(right)].copy()

def classify(model, graph, sess, im):
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

    im = cv2.flip(im, 1)

    # Reshape
    res = cv2.resize(im, (128,128), interpolation=cv2.INTER_AREA)

    # Convert to float values between 0. and 1.
    res = res.astype(dtype="float64")
    res = res / 255
    res = np.reshape(res, (1, 128, 128, 3))

    with graph.as_default():
        with sess.as_default():
            prediction= model.predict(res)

    return prediction[0] 

class VideoCamera(object):
    def __init__(self):
        # initialize the video camera stream and read the first frame
        # from the stream
        
        self.video = cv2.VideoCapture(0)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 180)
        (self.grabbed, self.frame) = self.video.read()

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

        self.detection_graph, self.sess = load_inference_graph()

        print("***************************LOADING MODEL***************************")
        self.model, self.classification_graph, self.session = load_KerasGraph(PATH_TO_CNN_MODEL)

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.video.read()

    def __del__(self):
        #releasing camera
        self.video.release()
              
    def get_frame(self):
       #extracting frames
        im_width, im_height = (self.video.get(3), self.video.get(4))
        ret, frame = self.video.read()
                             
        image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_np = cv2.flip(image_np,1)
        
        boxes, scores = detect_objects(image_np, self.detection_graph, self.sess)

        score_thresh = 0.30
        
        num_hands_detect = 1

        res = get_box_image(num_hands_detect, score_thresh, scores, boxes, im_width, im_height, image_np)

        draw_box_on_image(num_hands_detect, score_thresh, scores, boxes, im_width, im_height, image_np)

        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        ret, jpeg = cv2.imencode('.jpg', image_np)

        predictions = []
        
        self.res = res
        
        # response = {
        #     'prediction': {
        #         'next': str(format(prediction[0][0], '.9f')),
        #         'start': str(format(prediction[0][1], '.9f')),
        #         'stop': str(format(prediction[0][2], '.9f')),
        #         'predicted_class': classes[np.argmax(prediction[0])]
        #     }
        # }

        return jpeg.tobytes()
        # return predictions

    def predict(self):
        if self.res is not None:
            class_res = classify(self.model, self.classification_graph, self.session, self.res)
            print(class_res)
            index = np.argmax(class_res)
            prediction = CLASSES[index]
            print("Predicted: " + prediction)
            # predictions.append(np.argmax(class_res))
            return index

    def read(self):
        # return the frame most recently read
        return self.frame

    def size(self):
        # return size of the capture device
        return self.video.get(3), self.video.get(4)

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
