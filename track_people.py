import cv2
import numpy as np
#import matplotlib.pyplot as plt
import datetime

cap = cv2.VideoCapture(0)

################################################

##### taking input point ######################

_, inp_img = cap.read()
inp_img = cv2.flip(inp_img, 1)
inp_img = cv2.blur(inp_img, (4,4))
gray_inp_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2GRAY)

##############################################

########## tracking starts here ###########


old_pts = np.array([[350, 180], [350, 350]], dtype=np.float32).reshape(-1,1,2)

backup = old_pts.copy()
backup_img = gray_inp_img.copy()

#### text o/p window
outp = np.zeros((480,640,3))

#### variable ####
ytest_pos = 40
###############

while True:
    _, new_inp_img = cap.read()
    new_inp_img = cv2.flip(new_inp_img, 1)
    new_inp_img = cv2.blur(new_inp_img, (4,4))
    new_gray = cv2.cvtColor(new_inp_img, cv2.COLOR_BGR2GRAY)     
    new_pts,status,err = cv2.calcOpticalFlowPyrLK(gray_inp_img, 
                         new_gray, 
                         old_pts, 
                         None, maxLevel=1,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                                         15, 0.08))
    #### boundries
    if new_pts.ravel()[0]  >= 600:
        new_pts.ravel()[0] = 600
    if new_pts.ravel()[1] >= 350:
        new_pts.ravel()[1] = 350
    if new_pts.ravel()[0]  <= 20:
        new_pts.ravel()[0] = 20
    if new_pts.ravel()[1] <= 150:
        new_pts.ravel()[1] = 150
    if new_pts.ravel()[2]  >= 600:
        new_pts.ravel()[2] = 600
    if new_pts.ravel()[3] >= 350:
        new_pts.ravel()[3] = 350
    if new_pts.ravel()[2]  <= 20:
        new_pts.ravel()[2] = 20
    if new_pts.ravel()[3] <= 150:
        new_pts.ravel()[3] = 150
    ###############

    ##### drawing line
    x,y = new_pts[0,:,:].ravel()
    a,b = new_pts[1,:,:].ravel()
    cv2.line(new_inp_img, (int(x),int(y)), (int(a),int(b)), (0,0,255), 15)
    

    cv2.imshow("ouput", new_inp_img)
    

    
    if new_pts.ravel()[0]  > 400 or new_pts.ravel()[2]  > 400:        
        if new_pts.ravel()[0] > 550 or new_pts.ravel()[2]  > 550:
            new_pts = backup.copy()
            new_inp_img = backup_img.copy()
            ytest_pos += 40
            cv2.putText(outp, "gone at {}".format(datetime.datetime.now().strftime("%H:%M")), (10,ytest_pos), 
                cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0))



        
    elif new_pts.ravel()[0]  < 200 or new_pts.ravel()[2]  < 200:        
        if new_pts.ravel()[0] < 50 or new_pts.ravel()[2]  < 50:
            new_pts = backup.copy()
            new_inp_img = backup_img.copy()
            ytest_pos += 40
            cv2.putText(outp, "came at {}".format(datetime.datetime.now().strftime("%H:%M")), (10,ytest_pos), 
                cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255))
  
        
    
    cv2.imshow('final', outp)
    gray_inp_img = new_gray.copy()
    old_pts = new_pts.reshape(-1,1,2)

    if cv2.waitKey(1) & 0xff == 27:
        break
    
#### thank you for this video

cv2.destroyAllWindows()
cap.release()
