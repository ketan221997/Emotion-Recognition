'''
___________________________________________________________________________________________________________
Steps to be taken:
1) Connect to IP camera: https://stackoverflow.com/questions/49978705/access-ip-camera-in-python-opencv
2) Build a basic GUI using Tkinter
3) Convert to exe

Convert to exe once IP camera is connected and GUI is built.
To convert to exe use any of the following:
1) cx_freeze
2) py2exe
3) pyinstaller (personally recommended)

For a basic GUI use Tkinter, should not take longer than 2-3 hours.
____________________________________________________________________________________________________________
'''



'''
Importing the necessary libraries
cv2 - opencv for computer vision
numpy - mathematical library
keras - for neural network
warnings - to mute unecessary warnings
'''
import cv2
import numpy as np
from keras.models import load_model
from statistics import mode
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input
import warnings

warnings.filterwarnings("ignore")                   #Filter out warnings

                                                    
emotion_model_path = './models/emotion_model.hdf5'  #Load the neural network using the hdf5 file
emotion_labels = get_labels('fer2013')              #The different emotions- neutral, happy, angry, sad, surprise

frame_window = 10                                   #emotion needs to persist for 10 frames to be counted as valid
emotion_offsets = (20, 40)


print("[INFO] loading model...")                    #Load the DNN for face detection
net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")


#face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
emotion_classifier = load_model(emotion_model_path)


emotion_target_size = emotion_classifier.input_shape[1:3]   #Gets the input shape for the network which is 48 pixels
    
# starting lists for calculating modes
emotion_window = []


cv2.namedWindow('window_frame')
'''

Refer https://stackoverflow.com/questions/49978705/access-ip-camera-in-python-opencv
for accessing IP camera feed.

'''

cap = cv2.VideoCapture(0)                           #For accessing the camera, this field will need to be changed to make it work with an IP camera

#While the camera is open
while cap.isOpened():
    ret, bgr_image = cap.read()

    
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)#Grayscale the image and change the color space
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    (h, w) = rgb_image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(rgb_image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))#For detecting faces

    print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()                      #Gets the location of all detections

    '''loop over the detections'''
    for i in range(0, detections.shape[2]):         #For each face
        confidence = detections[0, 0, i, 2]         #Get the confidence level

        '''If we are more than 50 % sure that the object is a face then proceed with the detection'''
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")    #Co-ordinates of the face

            y = y1 - 10 if y1 - 10 > 10 else y1 + 10        

            gray_face = gray_image[y1:y2, x1:x2]    #Get the face
            try:
                gray_face = cv2.resize(gray_face, (emotion_target_size))    #Resize face to be fed to the neural network
            except:
                continue

            gray_face = preprocess_input(gray_face, True)   #Preprocess the face, such as getting rid of irrelevant areas like ears..   
            gray_face = np.expand_dims(gray_face, 0)        #Preprocessing before processing
            gray_face = np.expand_dims(gray_face, -1)
            
            emotion_prediction = emotion_classifier.predict(gray_face)#Getting predicted emotion from the NN
            
            emotion_probability = np.max(emotion_prediction)            #Getting the emotion with the maximum score/confidence
            emotion_label_arg = np.argmax(emotion_prediction)           #Getting corresponding emotion text - eg angry, happy
            emotion_text = emotion_labels[emotion_label_arg]
            emotion_window.append(emotion_text)

            if len(emotion_window) > frame_window:          #If the emotion is stable and not a temporary impulse eg during talking
                emotion_window.pop(0)
            try:
                emotion_mode = mode(emotion_window)
            except:
                continue

            if emotion_text == "angry" and emotion_probability < 0.50:  #As angry isn't that good, make sure that confidence in prediction is 
                emotion_text = "neutral"                                #at least 50 %
            elif emotion_text != 'happy' and emotion_text != "angry":   #Disabling other emotions and making all of them neutral as we don't want
                emotion_text = 'neutral'                                #poor predictions
            '''
             To test how the model works with other emotions, comment the last 2 lines, right above this comment.
            '''
            print(emotion_probability, " is the probability that the person is ", emotion_text)

            '''based on emotion, decide the color Red for angry, Yellow for Neutral and Green for Happy'''
            if emotion_text == 'angry':
                color = emotion_probability * np.asarray((255, 0, 0))
            elif emotion_text == 'sad':
                color = emotion_probability * np.asarray((0, 0, 255))
            elif emotion_text == 'happy':
                color = emotion_probability * np.asarray((0, 255, 0))
            elif emotion_text == 'surprise':
                color = emotion_probability * np.asarray((0, 255, 255))
            else:
                color = emotion_probability * np.asarray((255, 255, 0))

            color = color.astype(int)
            color = color.tolist()

            x_offset=y_offset = 100
            print(emotion_text)
            dic = {'happy' : "smiling.png", 'neutral' : "neutrall.png", 'angry' : "angry.png"} #Corresponding emojis for the particular emotion

            s_img = cv2.imread(dic[emotion_text], -1)                   #Read in the emoji pic
            s_img = cv2.resize(s_img, (50, 50))                         #Resize emoji
            
            sz = 3
##            if emotion_text == 'neutral':
##                sz -= 1
            alpha_s = s_img[:, :, sz] / 255.0
            alpha_l = 1.0 - alpha_s

            cv2.resize(rgb_image, (360, 480))
##            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)#Yeti

            y11 = max(0, y1)
            y22 = y11 + s_img.shape[0]
            x11 = max(0, x1- 40)
            x22 = x11 + s_img.shape[1]                                  #Co ordinates of the emoji 
            
            s_img = cv2.cvtColor(s_img, cv2.COLOR_BGR2RGB)

            '''This for loop overlaps the emoji and the current frame'''
            for c in range(3):
                rgb_image[y11:y22, x11:x22, c] = (alpha_s * s_img[:, :, c] +
                                          alpha_l * rgb_image[y11:y22, x11:x22, c])
            
            draw_bounding_box([x1, y1, x2 - x1, y2 - y1], rgb_image, color)#drawing a bounding box

        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)          #Convert back to right color space
        cv2.imshow('window_frame', cv2.resize(bgr_image, (520, 480)))   #Display the frame
        
        if cv2.waitKey(1) & 0xFF == ord('q'):                           #Pressing q releases the camera
            break

cap.release()
cv2.destroyAllWindows()

'''
If nothing works, just pray.
'''
