import numpy
import cv2
import math

# An image size WxH is a H*W*3 matrix

# input: bgr image
# output: bgr enhanced image
def PhotoEnhancer(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # print type(hsv), hsv.shape ,hsv
    h, s, v = cv2.split(hsv)
    print 'v.shape', v.shape
    luminance = v
    base, detail, faces_pos = BilateralFilteringAndFaceFinding(luminance)
    # faces_pos = [(0, 0, 516, 516)] 
    # faces_pos = [(0, 0, 630, 808)]
    luminance = FaceBeautification(luminance, base, detail, faces_pos)
    luminance = ColorEnhancement(luminance, base, detail)
    v = luminance
    hsv = cv2.merge([h, s, v])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr
    
def FastBilateralFilter(img):
    # TODO: this is traditional implement, need fast implement
    
    # Diameter of each pixel neighborhood that is used during filtering.
    # Large filters (d > 5) are very slow,
    # so it is recommended to use d=5 for real-time applications
    # and perhaps d=9 for offline applications that need heavy noise filtering
    diameter = 15
    # For simplicity, you can set the 2 sigma values to be the same.
    # If they are small (< 10), the filter will not have much effect
    # else if they are large (> 150), they will have a very strong effect
    sigmaColor = 250
    sigmaSpace = 250
    
    dst = cv2.bilateralFilter(img, diameter, sigmaColor, sigmaSpace)
    return dst

FaceDetector = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
# output: facesPos = [(x,y,w,h), ..]
# raise exception if img doesnt have a face
def FindFacesLienhart(img):
    # https://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html
    faces = FaceDetector.detectMultiScale(img, 1.3, 5)
    return faces
    
# extract a part of image as a sub matrix
# input: pos = (x1, y1, w, h)
def Extract(img, pos):
    x1, y1, w, h = pos
    x2, y2 = x1 + w, y1 + h
    part = img[y1:y2, x1:x2]
    return part
    
# output: base, detail, faces_pos
def BilateralFilteringAndFaceFinding(luminance):
    base = FastBilateralFilter(luminance)
    detail = luminance - base
    faces_pos = FindFacesLienhart(luminance)
    return base, detail, faces_pos
    
    
def FaceBeautification(luminance, base, detail, faces_pos):
    print 'nFaces', len(faces_pos)
    for facePos in faces_pos:
        face = Extract(luminance, facePos)
        # cv2.imshow("FaceBeautification face", face); cv2.waitKey()
        faceBase = Extract(base, facePos)
        # cv2.imshow("base", faceBase); cv2.waitKey()
        faceDetail = Extract(detail, facePos)
        # cv2.imshow("detail", faceDetail); cv2.waitKey()
        skinMap = FindSkinMap(face, 15.0, 15.0)
        # cv2.imshow("skinMap", skinMap); cv2.waitKey() 
        Smoothing(face, faceBase, faceDetail, skinMap)
        Blending(luminance, face, facePos, faceDetail)
    return luminance

# input: th_x th_y are the thresholds of smooth pixels
def FindSkinMap(face, th_x, th_y):
    sobel_x = cv2.Sobel(face, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(face, cv2.CV_64F, 0, 1)
    skin_map = face.copy()
    # cv2.imshow("b skinMap", sobel_x); cv2.waitKey() 
    faceHeight, faceWidth = face.shape
    for row in xrange(0, faceHeight):
        for col in xrange(0, faceWidth):
            x_value = sobel_x[row, col]
            y_value = sobel_y[row, col]
            temp = Sqr(th_x) +Sqr(th_y)
            # print 'x_value, y_value', x_value, y_value
            if (x_value < th_x and y_value < th_y and
                    Sqr(x_value) +Sqr(y_value) < temp):
                skin_map[row, col] = 255.0
            else:
                skin_map[row, col] = 0.0
    # print skin_map
    # cv2.imshow("a skinMap", skin_map); cv2.waitKey()             
    return skin_map

# calc square
def Sqr(x):
    return x * x

def Smoothing(face, faceBase, faceDetail, skin_map):
    alpha = 0
    faceHeight, faceWidth = face.shape
    # print 'faceDetail', faceDetail
    # cv2.imshow('Before Smoothing', face); cv2.waitKey()
    # cv2.imshow('faceBase', faceBase); cv2.waitKey()
    # cv2.imshow('faceDetail', faceDetail); cv2.waitKey()
    for row in xrange(0, faceHeight):
        for col in xrange(0, faceWidth):
            if skin_map[row, col] > 254:
                face[row, col] = faceBase[row, col] + alpha*faceDetail[row, col]
            else:
                pass
    # cv2.imshow('Smoothing', face); cv2.waitKey()  
        
def Blending(luminance, face, facePos, faceDetail):
    centerX, centerY = [d/2 for d in face.shape]
    facePosX, facePosY, faceW, faceH = facePos
    mask_map = numpy.zeros((luminance.shape))
    # print 'Blending', mask_map.shape, luminance.shape, faceDetail.shape
    for row in xrange(facePosY, facePosY+faceH):
        for col in xrange(facePosX, facePosX+faceW):
            # mask_map[row, col] = 1.0 - ((row- centerY) / (faceH/2.0) * (col-centerX)/ (faceW/2.0))
            luminance[row, col] = (face[row-facePosY, col-facePosX] +
                0 * (1.0 - mask_map[row, col])*faceDetail[row-facePosY, col-facePosX])
    # cv2.imshow('Blending', luminance); cv2.waitKey()          
        
def ColorEnhancement(luminance, base, detail):
    avg, sdv, lmax, lmin = CalcAvgSdvMaxMin(luminance)
    print 'avg, sdv, lmax, lmin', avg, sdv, lmax, lmin
    h, w = luminance.shape
    th = 15.0
    alpha = min(1.2, 70/sdv)
    beta = 0
    gamma = 1.1
    print 'alpha, beta, gamma', alpha, beta, gamma
    for row in xrange(0, h):
        for col in xrange(0, w):
            luminance[row, col] = Limit(luminance[row, col] * alpha + beta)
            luminance[row, col] = math.pow(luminance[row, col]/255.0, gamma) * 255
    return luminance

def CalcAvgSdvMaxMin(mat2d):
    max = 0
    min = 255
    h, w = mat2d.shape
    sum1 = 0.0
    for row in xrange(0, h):
        for col in xrange(0, w):
            sum1 += mat2d[row, col]
            if mat2d[row, col] > max:
                max = mat2d[row, col]
            if mat2d[row, col] < min:
                min = mat2d[row, col]
    avg = sum1 / (h*w)
    sum2 = 0.0
    for row in xrange(0, h):
        for col in xrange(0, w):
            sum2 += Sqr(mat2d[row, col] - avg)
    sdv = math.sqrt(sum2 / (h*w))
    return avg, sdv, max, min

def LogarithmicCurve(mat2d, row, col , beta):
    temp = math.log(mat2d[row,col] * (beta-1) + 1) / math.log(beta)
    mat2d[row,col] = beta ** temp
  
def Limit(n):
    if n <0:
        return 0
    elif n > 255:
        return 255
    else:
        return n
  
if __name__ == '__main__':
    imgPath = "./examples/a7.jpg"
    img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
    
    new = PhotoEnhancer(img)
    
    # cv2.imshow("final", new); cv2.waitKey()
    cv2.imwrite("./examples/o1.jpg", new)
    
    print 'xong'
