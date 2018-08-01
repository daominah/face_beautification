import main
import cv2

def Test01():
    cases = [
        ("./examples/a1.jpg", 2),
        ("./examples/a2.jpg", 0),
        ("./examples/a5.jpg", 1),
        ("./examples/a6.jpg", 1),
        ]
    for case in cases:
        imgPath = case[0] 
        img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
        # print 'img.shape', img.shape
        faces = main.FindFacesLienhart(img)
        if len(faces) == 0:
            pass
        else:
            faceMat = main.Extract(img, faces[0])
            # print 'faces[0]', faces[0], faceMat.shape
            # cv2.imshow('window name', faceMat) ; cv2.waitKey()
        if len(faces) != case[1]:
            raise Exception(len(faces), case[1])
        
    
def Test02():
    pass
    
Test01()
Test02()

print 'PASS'