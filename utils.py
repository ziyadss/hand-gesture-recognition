import matplotlib.pyplot as plt
import cv2


def show_cv2_image_bgr(img):
    print_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(print_img)
    plt.show()


def show_cv2_image_gray(img):
    plt.imshow(img, cmap="gray")
    plt.show()
