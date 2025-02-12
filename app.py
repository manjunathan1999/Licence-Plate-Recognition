import os
import re
import cv2
import pytesseract

SAVED = "./saved/"

class Licence_Recognize():

    def Licence_detect_main(rtsp):
        plateCascade = cv2.CascadeClassifier("./haarcascade_russian_plate_number.xml")
        minArea = 500
        license = Licence_Recognize()
        cap = cv2.VideoCapture(rtsp)
        cap.set(10,150)
        count = 0
        
        with open('./vocabulary.txt', 'r') as f:
            vocabulary = f.read().splitlines()
        
        while True:
            success , img  = cap.read()
            # print(success)
            imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            imgGray = cv2.medianBlur(imgGray, 3)
            _, imgGray = cv2.threshold(imgGray, 127, 255, cv2.THRESH_BINARY)
            numberPlates = plateCascade.detectMultiScale(imgGray, 1.1, 4)
            
            for (x, y, w, h) in numberPlates:
                area = w * h
                if area > minArea:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(img,"Number Plate Scanned",(x,y-5),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
                    imgRoi = img[y:y+h,x:x+w]
                    cv2.imshow("Number Plate",imgRoi)
                    
            img = cv2.resize(img, (700, 500))
            cv2.imshow("Video Frame",img)
            
            if cv2.waitKey(1) & 0xFF ==ord('s'):
                cv2.imwrite(SAVED+str(count)+".jpg",imgRoi)
                cv2.rectangle(img,(0,200),(640,300),(0,255,0),cv2.FILLED)
                cv2.putText(img,"Saved",(15,265),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),2)
                cv2.imshow("Result",img)
                cv2.waitKey(500)
                count==1
                saved_image_path = SAVED + "0.jpg"
                if os.path.exists(saved_image_path):
                    saved_image = cv2.imread(saved_image_path)
                    text = pytesseract.image_to_string(saved_image)
                    filtered_text = license.filter_text(text, vocabulary)
                    print("Detected text from saved image:", filtered_text)
                    
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        cap.release()
        cv2.destroyAllWindows()

    def filter_text(text, vocabulary):
        filtered_text = ''.join(char for char in text if char.isalnum() and char in vocabulary)
        re.match(r'^[A-Z]{2}[ -]?[0-9]{2}[ -]?[A-Z]{1,2}[ -]?[0-9]{4}$', filtered_text)
        return filtered_text
