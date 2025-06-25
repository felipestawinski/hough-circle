import cv2 as cv
import numpy as np

def main():

    
    img = cv.imread('img/bicicleta.jpg')
    if img is None:
        print("Error: Could not read the image.")
        return
    
    # Resize the image to 640x480
    img = cv.resize(img, (999, 666))
    
    #Detect edges using Canny
    edges = cv.Canny(img, 100, 200)
    cv.imshow("Edges", edges)
    
    
    cv.imshow("Original Image", img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    
    
    
if __name__ == "__main__":
    main()