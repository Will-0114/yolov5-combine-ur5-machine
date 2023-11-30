import cv2
import numpy as np

def create_black_img(row= 512, col =512, color = (0, 0, 0)):
    img = np.zeros((row, col, 3),np.uint8)
    return img

def draw_line(img, start_pt = (10, 10), end_pt = (100, 100), color = (0, 0, 255), lineWidth = 1):
    image = img.copy()
    image  = cv2.line(img, start_pt, end_pt, color, lineWidth)
    return image

def draw_rectangle(img, p1 =(100, 100), p2 = (200, 200), color = (0, 0, 255), lineWidth = 1):
    image = img.copy()
    cv2.rectangle(image, p1, p2, color, lineWidth) ## p1= (x, y) = (col, row)
    return image

def draw_circle(img, center = (100, 100), radius = 50, color = (0, 0, 255), lineWidth = 1):
    image = img.copy()
    cv2.circle(image, center, radius, color, lineWidth)
    return image

def draw_ellipse(img, center = (100, 100), axis =(100, 50), angle = 0, start_angle =0 , end_angle = 360, color = (0, 0, 255), lineWidth = 1):
    image = img.copy()
    cv2.ellipse(image, center, axis, angle, start_angle, end_angle, color, lineWidth) 
    return image

##def write_text(img, text = "OpenCV", position =(100, 100), color = (0, 0, 255), font = cv2.FONT_HERSHEY_PLAIN, fontScale = 1, lineType = cv2.LINE_AA, lineWidth = 1):  
##    image = img.copy()
##    ##cv2.putText(影像, 文字, 座標, 字型, 大小, 顏色, 線條寬度, 線條種類)
##    image  = cv2.putText(img, text, (150, 100), font, fontScale, color, lineWidth, lineType)
##    return image

def write_text(img, x, y, text, size = 16, color =(0, 255, 0, 0), font = cv2.FONT_HERSHEY_PLAIN, width =1):
        ## cv2.putText(影像, 文字, 座標, 字型, 大小, 顏色, 線條寬度, 線條種類)
        ##cv2.putText(img, text, (x, y), font, size, color, width, cv2.LINE_AA)
        ## cv cannot show non ASCII text, use pil to do that
        #color = (0,255,0,0) #(b,g,r,a) =   ## color setting
        from PIL import ImageFont, ImageDraw, Image
        fontpath = "./simsun.ttc" # <== 这里是宋体路径 
        font = ImageFont.truetype(fontpath, size)
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        draw.text((x, y),  text, fill = color, font = font )  # font = font (b, g, r, a)
        img = np.array(img_pil)
        return img

def main():
    img  =create_black_img()
    img = draw_line(img, (100, 100), (200, 200))
    img = draw_line(img, (100, 200), (200, 100))
    img = draw_rectangle(img, (100, 100), (200, 200))
    img = draw_circle(img, (150, 150), radius = 50)
    img = draw_ellipse(img, (150, 150), (100, 50), 0, 0, 360, color = (0, 255, 0))
    img = write_text(img, text = "Hello")
    
    cv2.imshow("black image", img)
    cv2.waitKey(-1)


if __name__ == "__main__":
    main()
