from ultralytics import YOLO
import cv2
import cvzone
import math
import time
from pass2 import ok
from email.message import EmailMessage
import ssl
import smtplib


def sendmails():
    a = ok()
    email_sender = "gaurav26999@gmail.com"
    email_password = a
    email_receiver = 'ng.niesh123@gmail.com'

    subject = ' Report of Drone Flying Near Border Area'
    body = """
   Dear sir,

I am writing to report that I have detected a drone flying near the border area. As you may be aware, drones can pose a security threat and may be used for illegal activities, which is why I thought it best to bring this to your attention.

The drone was spotted on [Date] at approximately [Time] in the vicinity of [Location], which is close to the border area. I was able to observe the drone for [Duration] before it disappeared from sight.

I am not sure who the operator of the drone was, but it appeared to be flying in a manner that suggested it may have been conducting surveillance. Given the sensitivity of the area, I thought it was important to report this to you immediately.

Please let me know if there is any further information that I can provide to assist with your investigation. I am happy to answer any questions you may have.

Thank you for your attention to this matter.

Sincerely,
[team xyz]




    """

    em = EmailMessage()
    em['From'] = email_sender
    em['To'] = email_receiver
    em['Subject'] = subject
    em.set_content(body)

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
        smtp.login(email_sender, email_password)
        smtp.sendmail(email_sender, email_receiver, em.as_string())




# cap =cv2.VideoCapture(0)
# cap.set(3, 1280)
# cap.set(4, 720)
k=int(0)
cap=cv2.VideoCapture("../Videos/drones8.mp4")


# cap = cv2.VideoCapture(1)  # For Webcam

# cap = cv2.VideoCapture("../Videos/drones6.mp4")  # For Video


model = YOLO("drones.pt")



classNames = ['dr', 'drone']

prev_frame_time = 0
new_frame_time = 0

while True:
    new_frame_time = time.time()
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            if ((classNames[cls]=="dr" or classNames[cls]=="drone") and (k == 0)):
                k = 1
                sendmails()

            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)



    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(fps)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

