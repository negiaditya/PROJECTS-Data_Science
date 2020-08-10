import matplotlib.pylab as plt
import cv2
import numpy as np

def region_of_interest(img,vertices):
	mask = np.zeros_like(img)
	#channel_count=img.shape[2]
	match_mask_color=  255   #(255,) * channel_count
	cv2.fillPoly(mask,vertices,match_mask_color)
	masked_image=cv2.bitwise_and(img,mask)
	return masked_image

def draw_the_lines(img,lines):
	imge=np.copy(img)
	blank_image=np.zeros((imge.shape[0],imge.shape[1],3),dtype=np.uint8)

	for line in lines:
		for x1,y1,x2,y2 in line:
			cv2.line(blank_image,(x1,y1),(x2,y2),(0,255,0),thickness=5)

	imge = cv2.addWeighted(imge,0.8,blank_image,1,0.0)
	return imge

def process(image):
	height=image.shape[0]
	width=image.shape[1]

	region_of_interest_coor = [ (0,height),(0,400), (width/2,height/3),(width,height)]

	gray_image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
	canny_image = cv2.Canny(gray_image,100,200)
	cropped_image = region_of_interest(canny_image,np.array([region_of_interest_coor], np.int32),)
	lines = cv2.HoughLinesP(cropped_image,
                            rho=2,
                            theta=np.pi/180,
                            threshold=120,
                            lines=np.array([]),
                            minLineLength=20,
                            maxLineGap=35)
	image_with_lines = draw_the_lines(image, lines)
	return image_with_lines

cap = cv2.VideoCapture('video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    frame = process(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()