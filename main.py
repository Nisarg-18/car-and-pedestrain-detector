import cv2

cascade_src = 'cars.xml'
video_src = 'video.mp4'
#video_src = 'video1.avi'
#video_src = 'video2.avi'

cap = cv2.VideoCapture(video_src)
car_cascade = cv2.CascadeClassifier(cascade_src)
pedestrian_tracker = cv2.CascadeClassifier('pedestrian.xml')

while True:
    ret, img = cap.read()
    if (type(img) == type(None)):
        break
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)

    pedestrians = pedestrian_tracker.detectMultiScale(gray)

    for (x,y,w,h) in cars:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)      
    
    for (x,y,w,h) in pedestrians:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)

    cv2.imshow('video', img)
    
    cv2.waitKey(1)

cv2.destroyAllWindows()