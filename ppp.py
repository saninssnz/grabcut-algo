import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(r"C:\Users\abdussamad p\PycharmProjects\pythonProject\venv\food20dataset\train_set\dosa\dosatrain (38).jpg")

print('Original Dimensions : ', img.shape)


width = 220
height = 220
dim = (width, height)
# resize image
resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

print('Resized Dimensions : ', resized.shape)

cv2.imshow("Resized image", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

#denoising

dst = cv2.fastNlMeansDenoisingColored(resized,None,10,10,7,21)

plt.subplot(121),plt.imshow(resized)
plt.subplot(122),plt.imshow(dst)
plt.show()

mask = np.zeros(dst.shape[:2], np.uint8)

bgdModel = np.zeros((1,65), np.float64)
fgdModel = np.zeros((1,65), np.float64)

rect =  (50,50,300,500)

cv2.grabCut(dst,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2) | (mask==0),0,1).astype('uint8')
dst = dst*mask2[:,:,np.newaxis]
plt.imshow(dst)
plt.colorbar()
plt.show()