#!/usr/bin/env python3
# Description:
# - Subscribes to real-time streaming video from your built-in webcam.
#
# Author:
# - Addison Sears-Collins
# - https://automaticaddison.com
 
# Import the necessary libraries
import rospy # Python library for ROS
from sensor_msgs.msg import Image # Image is the message type
from std_msgs.msg import String
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import cv2 # OpenCV library

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet

def face_rectangle(current_frame, x, y, w, h, confident_text):
  cv2.rectangle(current_frame, (x, y), (x + w, y + h), (255,0,255), 1)
  cv2.rectangle(current_frame, (x, y), (x + w, y + h), (255,0,255), 2)
  cv2.rectangle(current_frame, (x, y - 40), (x + w, y), (255,0,255), -1)

  cv2.putText(current_frame, confident_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
 
def callback(data):
 
  # Used to convert between ROS and OpenCV images
  br = CvBridge()
 
  # Output debugging information to the terminal
  rospy.loginfo("receiving video frame")
  # rospy.loginfo(a)
   
  # Convert ROS Image message to OpenCV image
  current_frame = br.imgmsg_to_cv2(data)

  rgb_img = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
  gray_img = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
  faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)
  for x,y,w,h in faces:
      img = rgb_img[y:y+h, x:x+w]
      img = cv2.resize(img, (160,160)) # 1x160x160x3
      img = np.expand_dims(img,axis=0)
      ypred = facenet.embeddings(img)
      face_name = model.predict(ypred)
      confident = model.decision_function(ypred)
      max_confident = round(max(confident[0]), 2)
      final_name = encoder.inverse_transform(face_name)[0]
      
      if max_confident >= 5.27:
        confident_text = str(final_name) + str(max_confident) + "%"
        face_rectangle(current_frame, x, y, w, h, confident_text)

        if final_name not in studentsInClassroom:
          studentsInClassroom.append(final_name)

          pub.publish(final_name)


      else :
        face_rectangle(current_frame, x, y, w, h, "unkhown")

  cv2.imshow("Face Recognition:", current_frame)
  # if cv2.waitKey(1) & 0xFF == ord('q'):
  #   break
  ###
   
  cv2.waitKey(1)
      
def receive_message():
 
  # Tells rospy the name of the node.
  # Anonymous = True makes sure the node has a unique name. Random
  # numbers are added to the end of the name. 
  rospy.init_node('video_sub_py', anonymous=True)


  # Node is subscribing to the video_frames topic

  rospy.Subscriber('video_frames', Image, callback)
  # rospy.Subscriber('~/camera/image/compressed', CompressedImage, callback)

  # rospy.Subscriber('video_frames', Image, callback)
 
  # spin() simply keeps python from exiting until this node is stopped
  rospy.spin()
 
  # Close down the video stream when done
  cv2.destroyAllWindows()
  
if __name__ == '__main__':
  pub = rospy.Publisher("name_detection", String, queue_size=10)
  studentsInClassroom = []

  ###
  facenet = FaceNet()

  faces_embeddings = np.load("/home/ubuntu-20-04/catkin_ws/src/cv_basics/scripts/data_set/faces_recognition_dataset.npz")
  Y = faces_embeddings['arr_1']
  encoder = LabelEncoder()
  encoder.fit(Y)
  haarcascade = cv2.CascadeClassifier("/home/ubuntu-20-04/catkin_ws/src/cv_basics/scripts/haarcascade_frontalface_default.xml")

  model = pickle.load(open("/home/ubuntu-20-04/catkin_ws/src/cv_basics/scripts/model/svm_dataset_model_160x160.pkl", 'rb'))
  ###
  receive_message()
