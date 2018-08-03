import cv2
import sys
import os.path
import os

class detect_face():
    def __init__(self):
        self.counter = 0

    def detect(self,filename, cascade_file = "./lbpcascade_animeface.xml"):
        if not os.path.isfile(cascade_file):
            raise RuntimeError("%s: not found" % cascade_file)
    
        cascade = cv2.CascadeClassifier(cascade_file)
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        faces = cascade.detectMultiScale(gray,
                                         # detector options
                                         scaleFactor = 1.1,
                                         minNeighbors = 5,
                                         minSize = (24, 24))
        for (x, y, w, h) in faces:
            #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            image_face_data = image[y:y+h,x:x+w] #切り取り
            image_face_data = cv2.resize(image_face_data,(224,224))
            #cv2.imshow("AnimeFaceDetect", image_face_data)
            #cv2.waitKey(0)
            file_name_data = "data" + str(self.counter) + ".png"
            self.counter +=1
            #dir_output_data = "./data_set/train_dataset/true_dataset/"
            dir_output_data = "./data_set/train_dataset/false_dataset/"
            full_dir_file_name = dir_output_data + file_name_data  
            print(full_dir_file_name)
            cv2.imwrite(full_dir_file_name, image_face_data)

def main():
    data_dir_path = "./rori_or_other/other_data"
    files = os.listdir('./rori_or_other/other_data')
    detector = detect_face()
    for file_name in files:
        root, ext = os.path.splitext(file_name)
        if ext == '.png' or 'jpeg' or '.jpg':
            abs_name = data_dir_path + '/' + file_name
            print(abs_name)
            detector.detect(abs_name)





if __name__ == '__main__':
    main()
