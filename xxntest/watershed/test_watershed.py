import cv2
import numpy as np

img = cv2.imread('coins.png')
# (780, 632, 3)
# print(img.shape)
# 将彩色图像转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# (780, 632)
# print(gray.shape)
# numpy
# print(type(gray))

# print("数组中的唯一值：", np.unique(gray))
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# 变成2值图像了
# print("数组中的唯一值：", np.unique(thresh))
cv2.imwrite('coins_thresh.png', thresh)
# noise removal
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

sure_bg = cv2.dilate(opening, kernel, iterations=2)  # sure background area
sure_fg = cv2.erode(opening, kernel, iterations=2)  # sure foreground area
unknown = cv2.subtract(sure_bg, sure_fg)  # unknown area
cv2.imwrite('coins_sure_bg.png', sure_bg)
cv2.imwrite('coins_sure_fg.png', sure_fg)
cv2.imwrite('coins_unknown.png', unknown)
# Perform the distance transform algorithm
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
# Normalize the distance image for range = {0.0, 1.0}
cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)

# Finding sure foreground area
ret, sure_fg = cv2.threshold(dist_transform, 0.5*dist_transform.max(), 255, 0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
cv2.imwrite('coins_unknown1.png', unknown)
# print("数组中的唯一值：", np.unique(unknown))
# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0
markers_copy = markers.copy()
markers_copy[markers==0] = 150  # 灰色表示背景
markers_copy[markers==1] = 0    # 黑色表示背景
markers_copy[markers>1] = 255   # 白色表示前景

markers_copy = np.uint8(markers_copy)
# 使用分水岭算法执行基于标记的图像分割，将图像中的对象与背景分离
markers = cv2.watershed(img, markers)
img[markers==-1] = [0,0,255]  # 将边界标记为红色
# 到这里就放弃了
cv2.imwrite('ans.png', img)