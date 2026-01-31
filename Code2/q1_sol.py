import numpy as np
import matplotlib.pyplot as plt
import cv2

# Question 1
#section a
inigo = cv2.imread('Inigo.jpg',cv2.IMREAD_GRAYSCALE).astype(np.float32)

delta = np.array([[0,0,0],[0,1,0],[0,0,0]],dtype=np.float32)
laplacian = np.array([[0,1,0],[1,-4,1],[0,1,0]],dtype=np.float32)
a_values = [0,0.5,0.75,1]
inigos = []
for a in a_values:
    filter = delta-a*laplacian
    new_inigo = cv2.filter2D(inigo,cv2.CV_32F,filter)
    new_inigo = np.clip(new_inigo,0,255).astype(np.uint8)
    inigos.append(new_inigo)


titles = ["a=0","a=0.5","a=0.75","a=1"]
fig,axes=plt.subplots(1,4,figsize=(30, 7))
for i,axis in enumerate(axes):
    axis.imshow(inigos[i],cmap='gray')
    axis.set_title(titles[i])
    axis.axis('off')
    axis.grid(False)
plt.tight_layout()
plt.show()

# section b

a_values_2 = [5,10]
inigos_2 = []
for a in a_values_2:
    filter = delta-a*laplacian
    new_inigo = cv2.filter2D(inigo,cv2.CV_32F,filter)
    new_inigo = np.clip(new_inigo,0,255).astype(np.uint8)
    inigos_2.append(new_inigo)

titles_2 = ["a=5","a=10"]
fig,axes=plt.subplots(1,2,figsize=(30, 7))
for i,axis in enumerate(axes):
    axis.imshow(inigos_2[i],cmap='gray')
    axis.set_title(titles_2[i])
    axis.axis('off')
    axis.grid(False)
plt.tight_layout()
plt.show()


# section c
a_values_3= [0.2,0.7]
inigos_3 = []
sp_inigo = inigo.copy()
density_mat = np.random.random(sp_inigo.shape) # density matrix of uniform probabilities
sp_inigo[density_mat<=0.02]=0 #0.02 will be black
sp_inigo[density_mat>0.98]=255 #0.02 will be white, so all in all 0.04 salt and pepper noise.

for a in a_values_3:
    filter = delta-a*laplacian
    new_inigo = cv2.filter2D(sp_inigo,cv2.CV_32F,filter)
    new_inigo = np.clip(new_inigo,0,255).astype(np.uint8)
    inigos_3.append(new_inigo)

titles_3 = ["a=0.2","a=0.7"]
fig,axes=plt.subplots(1,2,figsize=(30, 7))
for i,axis in enumerate(axes):
    axis.imshow(inigos_3[i],cmap='gray')
    axis.set_title(titles_3[i])
    axis.axis('off')
    axis.grid(False)
plt.tight_layout()
plt.show()

# section d

a_values_4= [0.2,0.7]
inigos_4 = []
sp_inigo_2 = inigo.copy()
density_mat_2 = np.random.random(sp_inigo_2.shape) # density matrix of uniform probabilities
sp_inigo_2[density_mat_2<=0.02]=0 #0.02 will be black
sp_inigo_2[density_mat_2>0.98]=255 #0.02 will be white, so all in all 0.04 salt and pepper noise.
# now we remove the salt and pepper noise
denoised_inigo= cv2.medianBlur(sp_inigo, 3)
for a in a_values_4:
    filter = delta-a*laplacian
    new_inigo = cv2.filter2D(denoised_inigo,cv2.CV_32F,filter)
    new_inigo = np.clip(new_inigo,0,255).astype(np.uint8)
    inigos_4.append(new_inigo)

titles_4 = ["a=0.2","a=0.7"]
fig,axes=plt.subplots(1,2,figsize=(30, 7))
for i,axis in enumerate(axes):
    axis.imshow(inigos_4[i],cmap='gray')
    axis.set_title(titles_4[i])
    axis.axis('off')
    axis.grid(False)
plt.tight_layout()
plt.show()

#section e
#repeat of section c:
a_values_5= [0.2,0.7]
inigos_5 = []
sp_inigo_3 = inigo.copy().astype(np.float32)
sp_inigo_3 /=255.0
sp_inigo_3 = np.random.poisson(sp_inigo_3 *40)/40
sp_inigo_3 = np.clip(sp_inigo_3*255, 0, 255).astype(np.uint8)

for a in a_values_5:
    filter = delta-a*laplacian
    new_inigo = cv2.filter2D(sp_inigo_3,cv2.CV_32F,filter)
    new_inigo = np.clip(new_inigo,0,255).astype(np.uint8)
    inigos_5.append(new_inigo)

titles_5 = ["a=0.2","a=0.7"]
fig,axes=plt.subplots(1,2,figsize=(30, 7))
for i,axis in enumerate(axes):
    axis.imshow(inigos_5[i],cmap='gray')
    axis.set_title(titles_5[i])
    axis.axis('off')
    axis.grid(False)
plt.tight_layout()
plt.show()

# repeat of section d:

a_values_6= [0.2,0.7]
inigos_6 = []
sp_inigo_6 = inigo.copy().astype(np.float32)
sp_inigo_6 /=255.0
sp_inigo_6 = np.random.poisson(sp_inigo_6 *40)/40
sp_inigo_6 = np.clip(sp_inigo_6*255, 0, 255).astype(np.uint8)

# now we remove the shot noise
kernel =(3,3)
denoised_inigo_2=cv2.fastNlMeansDenoising(sp_inigo_6, None,h=22,templateWindowSize=9,searchWindowSize=31)
for a in a_values_6:
    filter = delta-a*laplacian
    new_inigo = cv2.filter2D(denoised_inigo_2,cv2.CV_32F,filter)
    new_inigo = np.clip(new_inigo,0,255).astype(np.uint8)
    inigos_6.append(new_inigo)

titles_6 = ["a=0.2","a=0.7"]
fig,axes=plt.subplots(1,2,figsize=(30, 7))
for i,axis in enumerate(axes):
    axis.imshow(inigos_6[i],cmap='gray')
    axis.set_title(titles_6[i])
    axis.axis('off')
    axis.grid(False)
plt.tight_layout()
plt.show()
