import cv2
import numpy as np
import matplotlib.pyplot as plt

#imread(): to load the image || imshow():to display the image


def make_coodinates(image,line_parameters):
    slope,intercept=line_parameters
    # print(image.shape)
    y1=image.shape[0]
    y2=int(y1*(3/5))
    x1=int((y1-intercept)/slope)
    x2=int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])



def average_slope_intercept(image,lines):
    left_fit=[]
    right_fit=[]
    for line in lines:
        x1,y1,x2,y2=line.reshape(4)
        parameters=np.polyfit((x1,x2),(y1,y2),1)
        # print(parameters)
        slope=parameters[0]
        intercept=parameters[1]
        if slope<0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))  
    # print(left_fit)
    # print(right_fit)    
    left_fit_average=np.average(left_fit,axis=0)  
    right_fit_average=np.average(right_fit,axis=0)  
    # print(left_fit_average,'left')
    # print(right_fit_average,'right')
    left_line=make_coodinates(image,left_fit_average)
    right_line=make_coodinates(image,right_fit_average)
    return np.array([left_line,right_line])



def canny(image):
    #Canny edge detection technique
    #conversion in grayscale -- makes process faster as only one colour so it can take one value
    #Finding the Grayscale Conversion
    gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) #RGB to grayscale Conversion


    #Reduce Noise-Smoothened the Noise
    #Averge value of the pixel round up--blurring
    blur=cv2.GaussianBlur(gray,(5,5),0)#5*5 kernel and deviation=0

    #Canny- small gradient and Strong gradient --Indentifies the change with the help of the gradient
    canny=cv2.Canny(blur,50,150) #Canny(image,low_threshold,high_threshold)

    return canny

#triangular Region 
def region_of_interest(image):
    height=image.shape[0] #height start from 0 ie y-axis
    polygons=np.array([[(200,height),(1100,height),(550,250) ]])
    mask=np.zeros_like(image) #Return an array of zeros with the same shape and type as a given array
                            #intensity
    cv2.fillPoly(mask,polygons,255) #fill our triangle completely white
    masked_image=cv2.bitwise_and(image,mask)
    return masked_image
 


def display_lines(image,lines):
    line_image=np.zeros_like(image)
    if lines is not None:
        # for line in lines:
        #     # print(line)
        #     x1,y1,x2,y2=line.reshape(4)
        #     cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
          for x1,y1,x2,y2 in lines:
            # print(line)
            # =line.reshape(4)                  RGB VALUE THICKNESS
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)#drawing the line segment connecting (x1,y1),(x2,y2)
    return line_image




# For image
# image=cv2.imread('C:\\Users\\PROSOFT\\Desktop\\AI_PROJECT\\Lane_Detection\\test_image.jpg') # multi-dimensional numpy array
# lane_image=np.copy(image)
# canny_image=canny(lane_image)
# cropped_image=region_of_interest(canny_image)
# lines=cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)   #rho=2 
# averaged_lines=average_slope_intercept(lane_image,lines)
# line_image=display_lines(lane_image,averaged_lines) #blending of images
# combo_image=cv2.addWeighted(lane_image,0.8,line_image,1,1)
# cv2.imshow("result",combo_image)#to decode the image
# cv2.waitKey(0) # displays the image for specified amout of milli seconds



#For video
# cap=cv2.VideoCapture("C:\\Users\\PROSOFT\\Desktop\\AI_PROJECT\\Lane_Detection\\test2.mp4")
# cap1= cv2.VideoCapture('C:\\Users\\PROSOFT\\Desktop\\AI_PROJECT\\Lane_Detection\\data_set\\frames\\videoplayback.mp4')

# while(cap.isOpened()):
#     # break;q
#     _,frame=cap.read()
#     canny_image=canny(frame)
#     cropped_image=region_of_interest(canny_image)
#     lines=cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
#     averaged_lines=average_slope_intercept(frame,lines)
#     line_image=display_lines(frame,averaged_lines)
#     combo_image=cv2.addWeighted(frame,0.8,line_image,1,1) 
#     cv2.imshow("result",combo_image)#to decode the image
#     if cv2.waitKey(1) & 0xFF==ord('q'):# displays the image for specified amout of milli seconds
#         break

# cap.release()
# cv2.destroyAllWindows()
# plt.imshow(canny)
# plt.show()

cap= cv2.VideoCapture('C:/Users/DELL/OneDrive/Desktop/AI_PROJECT/Lane_Detection/data_set/frames/videoplayback.mp4')

if (cap.isOpened()== False):
	print("Error opening video file")
while(cap.isOpened()):
	ret, frame = cap.read()
	if ret == True:
		cv2.imshow('', frame)
		if cv2.waitKey(25) & 0xFF == ord('q'):
			break
	else:
		break
cap.release()
cv2.destroyAllWindows()



