#!/usr/bin/env python3
# import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.ndimage
import numpy as np

# QUESTION 5 - TEMPLATE MATCHING
# section a:
large_num=99999999
def calc_SSD(image,template):
    img_double = np.asarray(image,dtype=np.float64) # the question tells us to use double which is float64.
    temp_double = np.asarray(template,dtype=np.float64)
    M=temp_double.shape[0]
    N=temp_double.shape[1]
    # according to formula we can expand the square as sumI^2-2*sumI*T+sumT^2
    # where sumT^2 is a constant (every entry is just T^2(i,j)), and 2*sum*I*T is 2D spatial correlation of T with I,
    # and sumI^2 is a 2D spatial correlation of I^2(x+i-M/2-1,y+j-N/2-1) with a kernel of all ones because the number one is neutral to multiplication.
    sumT2 = np.sum(temp_double**2)
    sumI2 = scipy.ndimage.correlate(img_double**2,np.ones(temp_double.shape))
    sumIT = scipy.ndimage.correlate(img_double,temp_double)

    S=sumT2+sumI2-2*sumIT
    # now we set the areas where the template doesn't fit to a large number
    S[-M//2:,:]=large_num
    S[:M//2,:]=large_num
    S[:,-N//2:]=large_num
    S[:,:N//2]=large_num
    return S

# section b:
poe_img = plt.imread('Text.jpg')
e10 = plt.imread('E10.jpg')
e11 = plt.imread('E11.jpg')
e12 = plt.imread('E12.jpg')
e14 = plt.imread('E14.jpg')
e16 = plt.imread('E16.jpg')
imgs = [e10,e11,e12,e14,e16]
titles = ["E10","E11","E12","E14","E16"]
fig,axes = plt.subplots(1,5,figsize=(20,6))
for i,axis in enumerate(axes):
    axis.imshow(imgs[i],cmap='gray')
    axis.set_title(titles[i])
    axis.axis('off')
plt.tight_layout()
plt.show()

# algorithm to find best match:
# we just find the template with the lowest SSD.
# however, we need to be careful, we must normalize the sizes of the templates because we need to sum the same amount of numbers for each template to compare reliably.
ssds=[]
for img in imgs:
    ssds.append(calc_SSD(poe_img,img))
sizes=[]
for img in imgs:
    sizes.append(img.size)
def best_match(list_ssds,list_sizes):
    minimum_list =[np.min(ssd) for ssd in list_ssds]
    result_list = [minimum_list[i]/list_sizes[i] for i in range(len(list_ssds))]
    return np.argmin(result_list)


best=best_match(ssds,sizes)
print(f"best match is {titles[best]}")


# section c

# Now, we load the 4 templates we cropped from the image and calculate SSDs
temp_a = plt.imread('../tmp_resources/tempa.jpg')
temp_A = plt.imread('../tmp_resources/tempacap.jpg')
temp_t = plt.imread('../tmp_resources/tempt.jpg')
temp_T = plt.imread('../tmp_resources/temptcap.jpg')
# we cropped the images with MSPaint and we saved as JPG so we need to convert to gray levels
temps = [temp_a,temp_A,temp_t,temp_T]
for i in range(len(temps)):
    temps[i] = np.dot(temps[i][...,:3],[0.299, 0.587, 0.114]) #this converts to grayscale

a_ssd = calc_SSD(poe_img,temps[0])
A_ssd = calc_SSD(poe_img,temps[1])
t_ssd = calc_SSD(poe_img,temps[2])
T_ssd = calc_SSD(poe_img,temps[3])

# let's normalize SSDs, so that we can get a unified treshold

a_ssd_nor = a_ssd/temp_a.size
A_ssd_nor = A_ssd/temp_A.size
t_ssd_nor = t_ssd/temp_t.size
T_ssd_nor = T_ssd/temp_T.size

# here are our templates:

titles = ["template a","template A","template t","template T"]
fig,axes = plt.subplots(1,4,figsize=(20,6))
for i,axis in enumerate(axes):
    axis.imshow(temps[i],cmap='gray')
    axis.set_title(titles[i])
    axis.axis('off')
plt.tight_layout()
plt.show()

# now we need to determine the threshold. for that, we plot the flattened and sorted SSD images without the large borders:

ssd_imgs = [np.sort(a_ssd_nor[a_ssd_nor<large_num/temp_a.size]),np.sort(A_ssd_nor[A_ssd_nor<large_num/temp_A.size]),np.sort(t_ssd_nor[t_ssd_nor<large_num/temp_t.size]),np.sort(T_ssd_nor[T_ssd_nor<large_num/temp_T.size])]
titles = ["a ssd","A ssd","t ssd","T ssd"]
fig,axes = plt.subplots(1,4,figsize=(30,7))
for i,axis in enumerate(axes):
    #let's zoom in only on the best matches. since the text is not long, I estimate every letter appears at most ~50 times.
    axis.plot(ssd_imgs[i][:50])
    axis.set_title(titles[i])
    axis.axis('on')
    axis.grid(True)
plt.tight_layout()
plt.show()

# from this, we see that the number of appearances is at the point where the slope becomes very high. this is because the error increases very much
# once it cannot find any more matches.
# so, once we find a spot where the derivative suddenly becomes very large, then the erorr increased because it didn't find more
# matches, then we know we found the number of occurences

derivative_a = np.diff(ssd_imgs[0])
derivative_A = np.diff(ssd_imgs[1])
derivative_t = np.diff(ssd_imgs[2])
derivative_T = np.diff(ssd_imgs[3])

# Let's plot the derivatives to determine threshold

derivatives = [derivative_a,derivative_A,derivative_t,derivative_T]
titles = ["a derivative","A derivative","t derivative","T derivative"]
fig,axes = plt.subplots(1,4,figsize=(30,7))
for i,axis in enumerate(axes):
    axis.plot(derivatives[i][:50])
    axis.set_title(titles[i])
    axis.axis('on')
    axis.grid(True)
plt.tight_layout()
plt.show()

threshold=270
#we add 1 to the result, because the derivative becomes very large at index i, which means the amount of numbers is i+1
occur_a = np.where(derivative_a>threshold)[0][0]+1
occur_A = np.where(derivative_A>threshold)[0][0]+1
occur_t = np.where(derivative_t>threshold)[0][0]+1
occur_T = np.where(derivative_T>threshold)[0][0]+1
print(f"Occurences of a: {occur_a}")
print(f"Occurences of A: {occur_A}")
print(f"Occurences of t: {occur_t}")
print(f"Occurences of T: {occur_T}")

# for the next part we load template of 'c' and template of 'k'
temp_c = plt.imread('c.jpg')
temp_k = plt.imread('k.jpg')
# let's calculate SSD of temp_c with the edgar allan poe image

ssd_c = calc_SSD(poe_img,temp_c)
c_ssd_nor = ssd_c/temp_c.size
ssd_c_adj = np.sort(c_ssd_nor[c_ssd_nor<large_num/temp_c.size])
#let's see derivative graph to find a threshold
derivative_c = np.diff(ssd_c_adj)


# Let's plot the derivatives to determine threshold

derivatives2 = [derivative_c]
titles2 = ["c derivative"]
fig,axis=plt.subplots(1,1,figsize=(30, 7))
axis.plot(derivatives2[0][:50])
axis.set_title(titles2[0])
axis.axis('on')
axis.grid(True)
plt.tight_layout()
plt.show()

#500 is a good threshold from this graph.
limit2=500
#to avoid loops as much as possible we will create a mask that finds pixels whose error is below limit2, to find all places where a 'c' is.
mask=(c_ssd_nor<limit2)
c_y,c_x = np.where(mask)
poe_img_new=poe_img.copy()
for i in range(len(c_y)): # place the template k instead of c for all coordinates
    y1=c_y[i]-temp_k.shape[0]//2
    y2=y1+temp_k.shape[0]
    x1=c_x[i]-temp_k.shape[1]//2
    x2=x1+temp_k.shape[1]
    poe_img_new[y1:y2,x1:x2]=temp_k
# display:
plt.figure(figsize=(20,7))
plt.imshow(poe_img_new,cmap='gray')
plt.title("Replaced c with k")
plt.axis('off')
plt.tight_layout()
plt.show()