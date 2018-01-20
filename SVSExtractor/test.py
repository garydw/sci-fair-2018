#import openslide
#import numpy as np
#import cv2
#s = openslide.OpenSlide(r"C:\Users\matthew\Downloads\TCGA-HC-7081-01A-01-TS1.6d85eaa2-aaa8-404c-8be1-f1be693b1792.svs")
#pil = s.associated_images['thumbnail'].convert('RGB')
#cv_im = np.array(pil)
#cv_im = cv2.cvtColor(cv_im, cv2.COLOR_RGB2BGR)
#avg_blur7 = cv2.blur(cv_im,(9,9))
#gray_scale = cv2.cvtColor(avg_blur7,cv2.COLOR_BGR2GRAY)
#ret, otsu = cv2.threshold(gray_scale, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
##cv2.imshow('image2', otsu)
#kernel = np.ones((5,5), np.uint8)
#dilated = cv2.dilate(otsu, kernel, iterations=5)
#cv2.imshow('img2', cv_im)
#cv2.imshow('img1', otsu)
#cv2.imshow('img', dilated)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#print("hello world")
svsutils