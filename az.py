# -*- coding: utf-8 -*-
"""
Created on Wed May  6 19:21:25 2020

@author: InKunAhn
"""

import cv2
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from scipy.spatial import distance as dist

#유료버전
subscription_key = 'fb80f93fc61c48ddb3a963d3f088c3a0'
#################################################
#face-detect 1
#무료

#face_api_url ='https://jihoon.cognitiveservices.azure.com/face/v1.0/detect'
#유료
face_api_url ='https://jhcash.cognitiveservices.azure.com/face/v1.0/detect'
find_similars_url='https://jhcash.cognitiveservices.azure.com/face/v1.0/findsimilars'

# Set the FACE_SUBSCRIPTION_KEY environment variable with your key as the value.
# This key will serve all examples in this document.
KEY = 'fb80f93fc61c48ddb3a963d3f088c3a0'

# Set the FACE_ENDPOINT environment variable with the endpoint from your Face service in Azure.
# This endpoint will be used in all examples in this quickstart.
ENDPOINT = 'https://jhcash.cognitiveservices.azure.com'

# Create an authenticated FaceClient.
face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

# Detect a face in an image that contains a single face
img_name = 'Jihoon.jpg'
image_data = open(img_name,'rb')
detected_faces = face_client.face.detect_with_stream(image=image_data)

if not detected_faces:
    raise Exception('No face detected from image {}'.format(img_name))

# Display the detected face ID in the first single-face image.
# Face IDs are used for comparison to faces (their IDs) detected in other images.
print('Detected face ID from', img_name, ':')

for face in detected_faces: 
    print (face.face_id)
    person1Id=face.face_id
    print()

# Save this ID for use in Find Similar
first_image_face_ID = detected_faces[0].face_id
    
    
video_capture = cv2.VideoCapture(0)
count=0

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    #gray = cv2.cvtColor(frame, cv2.IMREAD_COLOR)
    #data=np.array(gray).tobytes()
    #response = requests.post(face_api_url, headers=headers, params=parameters, data=,timeout=200)
    #전체프레임중에서 1/20만 들고 오는중
    if(int(video_capture.get(1)) % 1 == 0):
        print('Saved frame number : ' + str(int(video_capture.get(1))))
        cv2.imwrite("frame%d.jpg" % count, frame)
        print('Saved frame%d.jpg' % count)  
        image_data1 = open("frame%d.jpg" % count,'rb')
        detected_faces2 = face_client.face.detect_with_stream(image=image_data1,return_face_landmarks=True)
        for face in detected_faces2: 
            
            person2Id=face.face_id
            
            (x,y)=(face.face_landmarks.eye_left_bottom.x,face.face_landmarks.eye_left_bottom.y)
            (x1,y1)=(face.face_landmarks.eye_left_inner.x,face.face_landmarks.eye_left_inner.y)
            (x2,y2)=(face.face_landmarks.eye_left_outer.x,face.face_landmarks.eye_left_outer.y)
            (x3,y3)=(face.face_landmarks.eye_left_top.x,face.face_landmarks.eye_left_top.y)
            
            leftEye=dist.euclidean((x,y),(x3,y3))/dist.euclidean((x1,y1),(x2,y2))
            
            (x4,y4)=(face.face_landmarks.eye_right_bottom.x,face.face_landmarks.eye_right_bottom.y)
            (x5,y5)=(face.face_landmarks.eye_right_inner.x,face.face_landmarks.eye_right_inner.y)
            (x6,y6)=(face.face_landmarks.eye_right_outer.x,face.face_landmarks.eye_right_outer.y)
            (x7,y7)=(face.face_landmarks.eye_right_top.x,face.face_landmarks.eye_right_top.y)
            
            rightEye=dist.euclidean((x4,y4),(x7,y7))/dist.euclidean((x5,y5),(x6,y6))   
            bothEye=(leftEye+rightEye)*500
            
            print (face.face_id)
            print(bothEye)

        result=face_client.face.verify_face_to_face(person1Id, person2Id, person_group_id=None, 
                                            large_person_group_id=None, custom_headers=None, raw=False)
        print(result.is_identical,result.confidence)
   
        count += 1       


    
    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()