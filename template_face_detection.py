import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")

def catch_faces(frame):
    scale_factor = 1.3 # Quanto menor mais preciso, mais lento
    min_neighbors = 10  # Quanto mais vizinhos, maior a chance de ser um rosto

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.equalizeHist(gray_frame)

    faces = face_cascade.detectMultiScale(gray_frame, scale_factor, min_neighbors)
    result = []

    for face in faces:
        (x, y, w, h) = face
        face = gray_frame[y:y+h, x:x+w]
        result.append(face)

    return result

def detect_face(frame, template):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)  
    (h, w) = template.shape
   
    # threshold = 0.5
    # loc = np.where(result > threshold)
    # loc = list(zip(*loc[::-1]))

    # for point in loc:
    #     cv2.rectangle(frame, point, (point[0] + w, point[1] + h), (255, 0, 0))

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    cv2.rectangle(frame, max_loc, (max_loc[0] + w, max_loc[1] + h), (255, 0, 0))
    return frame

vid = cv2.VideoCapture(0)

faces = []

while (True):

    ret, frame = vid.read()

    if not faces: 
        faces = catch_faces(frame)
    
    for face in faces:
        frame = detect_face(frame, face)
    
    cv2.imshow("video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()