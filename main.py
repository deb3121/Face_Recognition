import cv2
face_cap = cv2.CascadeClassifier("C:/Users/lenovo/AppData/Local/Programs/Python/Python310/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml") #will capture the features of the face
video_cap = cv2.VideoCapture(0) #for capturing live video
while True :
    ret , video_data = video_cap.read() #I am capturing the video here 
    col = cv2.cvtColor(video_data,cv2.COLOR_BGR2GRAY) #converting the color to black and white to capture the features of our face (muscles) and then again colorising it 
    faces = face_cap.detectMultiScale(  #for covering the data of the face 
         col,
         scaleFactor=1.1,
         minNeighbors=5,
         minSize=(30, 30),
         flags=cv2.CASCADE_SCALE_IMAGE
    )
    for(x,y,w,h) in faces: #for making the box around the face 
        cv2.rectangle(video_data,(x,y),(x+w,y+h),(0,255,0),2) #to make the rectangle box with the parameters 
    cv2.imshow("Live_Video",video_data) # for displaying the box 
    if cv2.waitKey(10) == ord("a"): #for stopping the live video 
        break
video_cap.release()