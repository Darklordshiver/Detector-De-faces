import cv2
import numpy as np

# face = cv2.imread("./download.jpg")

# #Importando classicadore pre-treinados
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")

def detect_face(frame):
    scale_factor = 1.3 # Quanto menor mais preciso, mais lento
    eye_scale_factor = 1.1
    min_neighbors = 5  # Quanto mais vizinhos, maior a chance de ser um rosto

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.equalizeHist(gray_frame)

    faces = face_cascade.detectMultiScale(gray_frame, scale_factor, min_neighbors)
    result = frame[:]

    for face in faces:
        (x, y, w, h) = face
        cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0))
        
        faceROI = gray_frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(faceROI, eye_scale_factor, min_neighbors)

        for eye in eyes:
            (ex, ey, ew, eh) = eye
            cv2.rectangle(result, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 0, 0))

    return result

vid = cv2.VideoCapture(0)

while (True):

    ret, frame = vid.read()

    frame = detect_face(frame)
    cv2.imshow("video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()