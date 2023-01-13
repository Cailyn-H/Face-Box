import face_recognition as fr
import cv2

video = cv2.VideoCapture(0)

while True:
    ret,frame = video.read()

    convertedRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resizedFrame = cv2.resize(convertedRGB, (0, 0), fx=0.1, fy=0.1)

    if ret:
        locations = fr.face_locations(resizedFrame)
        faces = fr.face_encodings(resizedFrame,locations)
    ret = not ret

    for (top, left, bottom, right), face in zip(locations, faces):
        cv2.rectangle(frame,(left*10, top*10), (right*10, bottom*10), (255, 153, 255), 10)
    cv2.imshow('Window', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        video.release()
        cv2.destroyAllWindows()



