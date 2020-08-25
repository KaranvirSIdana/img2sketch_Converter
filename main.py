import cv2
import numpy as np

print(cv2.__version__)
img = "/home/karan/PycharmProjects/Invisibility_cloak/venv/img.jpg"
img = cv2.imread(img)

img_colorMapped = cv2.applyColorMap(img, cv2.COLOR_BGR2GRAY)
gray = cv2.cvtColor(cv2.UMat(img_colorMapped), cv2.COLOR_BGR2GRAY)  # converting an image into gray scale from RGB

gray = cv2.UMat.get(gray)
inverted = 255 - gray

blurred = cv2.GaussianBlur(inverted, (1, 1), 0.25)


def dodge(front, back):
    result = front * 255 / (255 - back)
    result[result > 255] = 255
    result[back == 255] = 255
    return result


final_img = dodge(blurred, gray)

# Create the identity filter, but with the 1 shifted to the right!
kernel = np.zeros((9, 9), np.float32)
kernel[4, 4] = 2.0  # Identity, times two!

# Create a box filter:
boxFilter = np.ones((9, 9), np.float32) / 81.0

# Subtract the two:
kernel = kernel - boxFilter

# Note that we are subject to overflow and underflow here...but I believe that
# filter2D clips top and bottom ranges on the output, plus you'd need a
# very bright or very dark pixel surrounded by the opposite type.

custom = cv2.filter2D(final_img, -1, kernel)

cv2.imshow('Original image', img)
cv2.imshow('gray', img_colorMapped)
cv2.imshow('Blurred image', blurred)
cv2.imshow('final_img', custom)

cv2.waitKey(0)
cv2.destroyAllWindows()
