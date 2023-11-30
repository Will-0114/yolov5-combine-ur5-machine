import cv2
import numpy as np
import random


def random_flip_img(img, horizontal_chance=0, vertical_chance=0):
    flip_horizontal = False
    if random.random() < horizontal_chance:
        flip_horizontal = True

    flip_vertical = False
    if random.random() < vertical_chance:
        flip_vertical = True

    if not flip_horizontal and not flip_vertical:
        return img

    flip_val = 1
    if flip_vertical:
        flip_val = -1 if flip_horizontal else 0

    if not isinstance(img, list):
        res = cv2.flip(img, flip_val) # 0 = X axis, 1 = Y axis,  -1 = both
    else:
        res = []
        for img_item in img:
            img_flip = cv2.flip(img_item, flip_val)
            res.append(img_flip)
    return res

def random_rotate_img(images):
    rand_roat = np.random.randint(4, size=1)
    angle = 90*rand_roat
    center = (images.shape[0] / 2, images.shape[1] / 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle[0], scale=1.0)

    img_inst = cv2.warpAffine(images, rot_matrix, dsize=images.shape[:2], borderMode=cv2.BORDER_CONSTANT)

    return img_inst

def random_crop(image, crop_size=(400, 400)):
    height, width = image.shape[:-1]
    dy, dx = crop_size
    X = np.copy(image)
    aX = np.zeros(tuple([3, 400, 400]))
    if width < dx or height < dy:
        return None
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    aX = X[y:(y + dy), x:(x + dx), :]
    return aX

def convert_cv2_2_pil(cvImg):
    # You may need to convert the color.
    img = cv2.cvtColor(cvImg, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    return im_pil

def draw_bbox_xywh(img, bboxs, color = (0, 255, 0), width = 2, isShow = True):
    img_bbox = img.copy()
    for f in bboxs: ## center is fearue[0]
        (x, y, w, h) = f[3]
        img_bbox = cv2.rectangle(img_bbox, (x, y), (x + w, y + h), color, width) ## -1 filled
    if isShow:
        cv2.imshow("image with bbox", img_bbox)
        cv2.waitKey(0)
    return img_bbox

def draw_bbox_xyxy(img, bboxs, color = (0, 255, 0), width = 2, isShow = True):
    img_bbox = img.copy()
    for x1, y1, x2, y2 in bboxs: ## center is fearue[0]
        img_bbox = cv2.rectangle(img_bbox, (int(x1), int(y1)), (int(x2), int(y2)), color, width) ## -1 filled
    if isShow:
        cv2.imshow("image with bbox", img_bbox)
        cv2.waitKey(0)
    return img_bbox

def cvRotate(img, degree = 180, isShow = True, pad_color = (0, 0, 0)):
    #img = cvImg.copy()
    if img is None:
        print("Image is empty")
        return None
    if len(img.shape)==3:
        rows, cols, channel = img.shape
    else:
        row, cols = img.shape
    center = (cols/2, rows/2)  ## 以中心旋轉
    scale = 1
    M = cv2.getRotationMatrix2D(center, degree, scale)
    #print("matrix", M)
    dst = cv2.warpAffine(img, M, (cols, rows), borderValue=pad_color)
    if isShow:
        cv2.imshow("rotated image", dst)
        cv2.waitKey(-1)
    return dst

def cvTrnaslate(img, translate = (5, 3), pad_color = (0, 0, 0), isShow = True): ## translate = (tx, ty)
    height, width = img.shape[:2] 
    #ty, tx = height / 2, width / 2
    ## build the matrix
    tx = translate[0]
    ty = translate[1]
    T = np.float32([[1, 0, tx], [0, 1, ty]]) ## translate matrix
    im_t = cv2.warpAffine(img, T, (width, height), borderValue=pad_color)  ## center (x, y)
    if isShow:
        cv2.imshow("Original", img)
        cv2.imshow("Tranlated", im_t)
        cv2.waitKey(0)
    return im_t

def cvImgPadding(img, pad_size = (10, 20), isShow = True):
    height, width = img.shape[:2] 
    ty, tx = height / 2, width / 2
    ## build the matrix
    px = pad_size[0]
    py = pad_size[1]
    T = np.float32([[1, 0, px], [0, 1, py]]) ## translate matrix
    im_pad = cv2.warpAffine(img, T, (height+px*2, width+py*2))
    if isShow:
        cv2.imshow("Original", img)
        cv2.imshow("Padding image", im_pad)
        cv2.waitKey(0)
    return im_pad

## CV2 Morphology
def cvColorErode(cvImg, kernel= None, iterations = 1, isShow=True):
    hsv = cv2.cvtColor(cvImg, cv2.COLOR_BGR2HSV) #convert it to hsv
    h, s, v = cv2.split(hsv)
    img_dilate = cv2.erode(v, kernel, iterations = 1)
    final_hsv = cv2.merge((h, s, img_dilate))
    erode_img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    if isShow:
        cv2.imshow("Orignal color image", cvImg)
        cv2.imshow("Color Erosion", erode_img)
        cv2.waitKey(0)
    return erode_img

def cvColorDilate(cvImg, kernel= None, iterations = 1, isShow=True):
    hsv = cv2.cvtColor(cvImg, cv2.COLOR_BGR2HSV) #convert it to hsv
    h, s, v = cv2.split(hsv)
    img_dilate = cv2.dilate(v, kernel, iterations = 1)
    final_hsv = cv2.merge((h, s, img_dilate))
    erode_img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    if isShow:
        cv2.imshow("Orignal color image", cvImg)
        cv2.imshow("Color Dilation", erode_img)
        cv2.waitKey(0)
    return erode_img
    
def cvDilate(img, kernel= None, iterations = 1, isShow=True):
    if kernel is None:
        kernel = np.ones((3,3),np.uint8)
    img_dilate = cv2.dilate(img, kernel, iterations = 1)
    if isShow:
        cv2.imshow("Orignal", img)
        cv2.imshow("Dilate", img_dilate)
    return img_dilate

def cvErode(img, kernel= None, iterations = 1, isShow=True):
    if kernel is None:
        kernel = np.ones((3,3),np.uint8)
    img_erode = cv2.erode(img, kernel, iterations = 1)
    if isShow:
        cv2.imshow("Orignal", img)
        cv2.imshow("Erode", img_erode)
    return img_erode

def cvThinning(img, isShow = True):
    if len(img.shape) ==3:
        cvImg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        cvImg = img.copy()
    thinned = cv2.ximgproc.thinning(cvImg)
    if isShow:
        cv2.imshow("GT", img)
        cv2.imshow("Thinning", thinned)
        cv2.waitKey(0)
    return thinned

def cvBrightned(cvImage, value): ## 每次修改intensity的數量
    hsv = cv2.cvtColor(cvImage, cv2.COLOR_BGR2HSV) #convert it to hsv
    #hsv[:,:,2] += value ## + ==> bright   - ==> dark
    # for x in range(0, len(hsv)):
    #     for y in range(0, len(hsv[0])):
    #         #print(hsv[x, y][2])
    #         if hsv[x, y][2] + value > 255:
    #             hsv[x, y][2] = 255
    #             #continue
    #         elif hsv[x, y][2] + value < 0:
    #             hsv[x, y][2] = 0
    #         else:
    #             hsv[x, y][2] += value
    # img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    h, s, v = cv2.split(hsv)
    #print(v.shape)
    rows, cols = v.shape
    #v_list = list(v)
    for i in range(rows):
        for j in range(cols):
            if v.item(i, j) == 255:  ## now background is 255
                    continue  ## background
            if value >=0:   
                lim = 255 - value
                if v.item(i, j) > lim:
                    v.itemset((i, j), 255)
                else:
                    v.itemset((i, j),  v.item(i, j) + value)
            else:
                #value = int(-value)
                lim = 0 + (-value)
                if v.item(i, j) < lim:
                    v.itemset((i, j), 0)
                else:
                    v.itemset((i, j),  v.item(i, j) + value)
    ### 以下是快速簡易寫法，但backgroud 會改變 ###
    # if value >= 0:
    #     lim = 255 - value
    #     v[v > lim] = 255
    #     v[v <= lim] += value
    # else:
    #     value = int(-value)
    #     lim = 0 + value
    #     v[v < lim] = 0
    #     v[v >= lim] -= value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def cvBlur(cvImage, mask= (3, 3)):
    img_blur = cv2.GaussianBlur(cvImage, mask, 0) # 0: 讓程式自動替計算
    #img_blur = cv2.blur(cvImage, mask)  ## average (mean filter)
    return img_blur

def cvSharpen(cvImage):  ## laplacian filter
    kernel = np.array([[-1,-1,-1], 
                   [-1, 9,-1],
                   [-1,-1,-1]])
    sharpened = cv2.filter2D(cvImage, -1, kernel) # applying the sharpening kernel to the input image & displaying it
    return sharpened

def cvRotate(img, degree = 180, isShow = True, pad_color = (0, 0, 0)):
    #img = cvImg.copy()
    if img is None:
        print("Image is empty")
        return None
    if len(img.shape)==3:
        rows, cols, channel = img.shape
    else:
        row, cols = img.shape
    center = (cols/2, rows/2)  ## 以中心旋轉
    scale = 1
    M = cv2.getRotationMatrix2D(center, degree, scale)
    #print("matrix", M)
    dst = cv2.warpAffine(img, M, (rows, cols), borderValue = pad_color)
    if isShow:
        cv2.imshow("rotated image", dst)
        cv2.waitKey(-1)
    return dst


def crop_Image(Img, x_start, y_start, x_end, y_end, isShow = False):
    cvImg = Img.copy()
    cropImg = cvImg[y_start: y_end, x_start: x_end]
    height, width, channel = cropImg.shape
    #print(height, width, channel)
    heading = "crop image"
    if isShow:
        #cv2.resizeWindow(heading, int(self.scale* self.width), int(self.scale* self.height))
        #cv2.namedWindow(heading, 0);
        cv2.imshow(heading, cropImg)
    return cropImg


def template_match_scale(img_color, template_color, method =0, threshold = 0.9, minScale = 0.5, maxScale = 1.5,  isShow = True):
    t_start = time.perf_counter()
    img_c = img_color.copy()
    if len(img_color.shape) == 3:
        img = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
    else:
        img = img_color.copy()
    
    if len(template_color.shape) == 3:
        template = cv2.cvtColor(template_color.copy(), cv2.COLOR_BGR2GRAY)
    else:
        template = template_color.copy()
    w, h = template.shape[::-1]
    # All the 6 methods for comparison in a list
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    #print("Matching method", methods[method])
    m = eval(methods[method])
    max = 0  ## max matching score
    max_scale =0
    max_index = 0 ## record the image index
    i = 0
    #res_list = list()
    max_res = None
    ## find the most similar scale
    for scale in np.arange(minScale, maxScale, 0.1):
        temp = cv2.resize(template.copy(), None, fx = scale, fy = scale)
        #print(temp.shape)
        res = cv2.matchTemplate(img, temp, method)

        if m == 5:  ## min is matched
            res = 1 - res
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val > max:
            max = max_val
            max_index = i  ## image index
            max_scale = scale
            max_res = res.copy()
            max_res_loc = max_loc
        i = i + 1
        #res_list.append(res)
    #print("Found the most similar scale: ", max_scale, max_index)
    ## find the location in max_index image
    map = np.where(res > threshold, 255, 0)
    loc = np.where( map >= threshold)
    #print(loc)
    map = np.array(map, dtype = np.uint8)
    map = cv2.cvtColor(map, cv2.COLOR_GRAY2BGR)
    ##map_thinned = cv2.ximgproc.thinning(map)  ## cannot find the center of the blob
    contours, img_contour = find_contour(map, threshold =127, isBlur = False, isShow = False, color = (0, 0, 255))
    feature_list = calc_contour_feature(img, contours, isShow = False)
    ## Note find centroid is not good, so try to find the max
    pt_list = list()
    for f in feature_list:
        box = f[3]
        box = [box[0], box[1], box[0]+box[2], box[1]+ box[3]]
        res_roi = crop_image(max_res, box)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res_roi)
        pt_list.append((max_loc[0] + box[0], max_loc[1]+box[1]))
    #print(pt_list)
    h = int(h* max_scale)
    w = int(w * max_scale)
    #print(h," : ", w)
    t_end = time.perf_counter()
    print("Matching Time: ", round((t_end - t_start), 3), " sec.")
    max_res = cv2.cvtColor(np.array(max_res*255, np.uint8), cv2.COLOR_GRAY2BGR)
    max_res_disp = max_res.copy()
    for pt in pt_list: #zip(*loc[::-1]): ## 配對位置 (x, y)
        cv2.rectangle(img_c, pt, (int(pt[0]) + w, int(pt[1]) + h), (0, 0, 255), 1)
        max_res_disp = draw_cross(max_res_disp, pt, isShow=False)
    if isShow:
        cv2.imshow(methods[m] + "Matching Score Matrix:", max_res_disp) #
        cv2.imshow(methods[m] + "Match Result:", img_c)
        cv2.waitKey(0)
    return img_c, max_res
## Example ##
#    img_rgb = cv2.imread('./images/Switch110.jpg', 1)
#    template_rgb = cv2.imread('./images/SwitchHole.bmp', 1)
#    template_match_scale(img_rgb, template_rgb, method = 1, threshold = threshold, minScale = 0.9, maxScale = 1.3, isShow = True)


## Crop image by bounding box (x1, y1, x2, y2)
def crop_image(image, bbox):
    image_crop = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    return image_crop

# 去掉太過接近的座標
def removeSame(pts, threshold): ## Threshold = distance
    elements = []
    for x,y in pts:
        for ele in elements:
            if ((x-ele[0])**2 + (y-ele[1])**2) < threshold**2:
                break
        else:
            elements.append((x,y))
    
    return elements

def template_match(img_color, template, method = 0, isShow = True):
    t_start = time.perf_counter()
    img_c = img_color.copy()
    img = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
    w, h = template.shape[::-1]
    # All the 6 methods for comparison in a list
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    #print("Matching method", methods[method])
    res = cv2.matchTemplate(img, template, method)
    #print(res)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    t_end = time.perf_counter()
    print("Matching Time: ", round((t_end - t_start), 3), " sec.")
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:  ## min is matched
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img_c, top_left, bottom_right, (255, 255, 0), 2)
    if isShow:
        cv2.imshow("Matching Score Matrix:", cv2.normalize(res, None, 0, 1, cv2.NORM_MINMAX)) #
        cv2.imshow("Match Result:", img_c)
        cv2.waitKey(0)
    return img_c, res
    
def template_match_multiple(img_color, template, method =0, threshold = 0.8, isShow = True):
    t_start = time.perf_counter()
    img_c = img_color.copy()
    img = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
    w, h = template.shape[::-1]
    # All the 6 methods for comparison in a list
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    #print("Matching method", methods[method])
    #m = eval(methods[method])
    res = cv2.matchTemplate(img, template, method)
    res = cv2.normalize(res, None, 0, 1, cv2.NORM_MINMAX)  ## 正規化
    
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:  ## min is matched
       res = 1 - res
       print(res)
    #min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    t_end = time.perf_counter()
    print("Matching Time: ", round((t_end - t_start), 3), " sec.")
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    loc = np.where( res >= threshold) ## loc: [[x1, x2, ...] [y1, y2, ....]]
    print(len(loc[0]))
    for pt in zip(*loc[::-1]): ## 配對位置 (x, y)
        cv2.rectangle(img_c, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 1)
    if isShow:
        cv2.imshow(methods[method] + "Matching Score Matrix:", res) #
        cv2.imshow(methods[method] + "Match Result:", img_c)
        cv2.waitKey(0)
    return img_c, res

def shape_match2(image, template, aspectThreshold = 0.1, keepAspectRatio = True, isBlack=True, method = 1, minNumContour =  10, isShow = True):
    ## input image must be grayscale
    ## template
    ## minNumContour: 有時候會有一些雜訊，需要過濾掉不的contour
    ## 預設: object 的aspect ratio 會保持不變，所以 ratio 差必須小於 0.1
    template_color = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
    ret, thresh = cv2.threshold(template, 127, 255, isBlack)
    contours, hierarchy= cv2.findContours(thresh, 2, 1)  # 所有內外contours
    cv2.drawContours(template_color, contours, -1, (0, 0, 255), 1)
    #cv2.waitKey(0)
    template_cnt = contours[0]
    (x,y),(w,h), angle = cv2.minAreaRect(template_cnt)
    if w > h:
        temp = w
        w = h
        h = temp
    template_aspect_ratio = float(w)/h
    #print("Template: ", w, h, template_aspect_ratio)
    ## detecting image
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    ret, thresh2 = cv2.threshold(image, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh2, 2, 1)
    image_disp = image_color.copy()
    cv2.drawContours(image_disp, contours, -1, (0, 0, 255), 1)
    print("Found contours: ", len(contours))
    color = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), 
                (255, 255, 0)]
    contour_list = list()
    score_list = list()
    for i, cont in enumerate(contours):
        if len(cont) < minNumContour:  ## ignor small object
            continue
        (x, y), (w, h), angle = rect = cv2.minAreaRect(cont)
        box = np.int0(cv2.boxPoints(rect))
        if w > h:
            temp = w
            w = h
            h = temp
        aspect_ratio = float(w)/h
        #print(aspect_ratio, template_aspect_ratio)
        
        if abs(aspect_ratio - template_aspect_ratio)>0.3:
            continue
        similiarity = cv2.matchShapes(template_cnt, cont, method, 0.0) ## method I, 0.0 none
        #print(similiarity)
        if similiarity < aspectThreshold: ## smaller is better
            print(w, h, aspect_ratio)
            M = cv2.moments(cont)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.drawContours(image_color, [box], 0, color[i%5], 1)
            cv2.putText(image_color, str(round(similiarity, 4)), (cX - 20, cY),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, color[i%5], 1)
            contour_list.append(cont)
            score_list.append(similiarity)
    if isShow:
        cv2.imshow("Template contour:", template_color)
        cv2.imshow("All contours", image_disp)
        cv2.imshow("Matched contours", image_color)
        cv2.waitKey(0)
    return contour_list, score_list, image_color
    
def find_contour(img, threshold =127, isBlur = True, isShow = True, color = (0, 255, 0)):
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if isBlur:
        imgray = cv2.GaussianBlur(imgray, (5, 5), 0)
    ## remember: object is white in a black background
    ret, thresh = cv2.threshold(imgray, threshold, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    color = color #[(0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 0, 0), (255, 255, 0),(0, 0, 255), (0, 255, 255), (255, 0, 255),]
    img_contour = img.copy()
    for i in range(len(contours)):
        cnt = contours[i]
        cv2.drawContours(img_contour, [cnt], 0, color, 1)
    #print(contours[0])
    #print(hierarchy)
    if isShow:
        cv2.imshow("Gray image", imgray)
        cv2.imshow("Threshold", thresh)
        cv2.imshow("contours", img_contour)
        cv2.imshow("image", img)
        cv2.waitKey(0)
    return contours, img_contour

def calc_contour_feature(img, contours, isShow = True):
    """
    輸入 contours
    回傳: feature list
    """
    feature_list = list()
    for cont in contours:
        area = cv2.contourArea(cont)
        if area == 0:
            continue
        perimeter = cv2.arcLength(cont, closed=True)
        bbox = cv2.boundingRect(cont)
        #print(bbox)
        bbox2 = cv2.minAreaRect(cont)
        #print(bbox2)
        circle = cv2.minEnclosingCircle(cont)
        if len(cont) > 5:
            ellipes = cv2.fitEllipse(cont)
        else:
            ellipes = None
        #print(ellipes)
        # Moment
        M = cv2.moments(cont) ## return all moment of given contour
        if area != 0: ## same as M["m00"] !=0
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            center = (cx, cy)
        else:
            center = (None, None)
        feature = (center, area, perimeter, bbox, bbox2, circle, ellipes)
        feature_list.append(feature)
    return feature_list

def draw_center(img, feature_list, color = (255, 255, 255), radius = 5, isShow = True):
    img_center = img.copy()
    for f in feature_list: ## center is fearue[0]
        if f[0][0] is not None:
            img_center = cv2.circle(img_center, f[0],  radius, color, -1) ## -1 filled
    if isShow:
        cv2.imshow("image with center", img_center)
        cv2.waitKey(0)
    return img_center


def draw_bbox(img, feature_list, color = (0, 255, 0), width = 2, isShow = True):
    img_bbox = img.copy()
    for f in feature_list: ## center is fearue[0]
        (x, y, w, h) = f[3]
        img_bbox = cv2.rectangle(img_bbox, (x, y), (x + w, y + h), color, width) ## -1 filled
    if isShow:
        cv2.imshow("image with bbox", img_bbox)
        cv2.waitKey(0)
    return img_bbox

def draw_bbox2(img, feature_list, color = (0, 255, 0), width = 2, isShow = True):
    img_bbox2 = img.copy()
    for f in feature_list: ## center is fearue[0]
        box = np.int0(cv2.boxPoints (f[4]))  #–> int0會省略小數點後方的數字
        img_bbox = cv2.drawContours(img_bbox2, [box], -1, color, width)
    if isShow:
        cv2.imshow("image with bbox", img_bbox2)
        cv2.waitKey(0)
    return img_bbox2

def draw_minSCircle(img, feature_list, color = (0, 255, 0), width = 2, isShow = True):
    img_circle = img.copy()
    for f in feature_list: ## center is fearue[0]
        ((x, y), radius) = f[5]  #–> int0會省略小數點後方的數字
        img_circle = cv2.circle(img_circle, (int(x), int(y)), int(radius), color, width)
    if isShow:
        cv2.imshow("image with bbox", img_circle)
        cv2.waitKey(0)
    return img_circle

## cv Drawing tools
def create_black_img(row= 512, col =512, color = (0, 0, 0)):
    img = np.zeros((row, col, 3),np.uint8)
    return img

def draw_line(img, start_pt = (10, 10), end_pt = (100, 100), color = (0, 0, 255), lineWidth = 1):
    image = img.copy()
    image  = cv2.line(img, start_pt, end_pt, color, lineWidth)
    return image

def draw_rectangle(img, p1 =(100, 100), p2 = (200, 200), color = (0, 0, 255), lineWidth = 1):
    image = img.copy()
    cv2.rectangle(image, p1, p2, color, lineWidth)
    return image

def draw_circle(img, center = (100, 100), radius = 50, color = (0, 0, 255), lineWidth = 1):
    image = img.copy()
    cv2.circle(image, center, radius, color, lineWidth)
    return image

def draw_ellipse(img, center = (100, 100), axis =(100, 50), angle = 0, start_angle =0 , end_angle = 360, color = (0, 0, 255), lineWidth = 1):
    image = img.copy()
    cv2.ellipse(image, center, axis, angle, start_angle, end_angle, color, lineWidth) 
    return image

def write_text(img, text = "OpenCV", position =(100, 100), color = (0, 0, 255), font = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, lineType = cv2.LINE_AA, lineWidth = 1):  
    image = img.copy()
    ##cv2.putText(影像, 文字, 座標, 字型, 大小, 顏色, 線條寬度, 線條種類)
    image  = cv2.putText(img, text, position , font, fontScale, color, lineWidth, lineType)
    return image

def main():
    img = cv2.imread('./Images/shapes_and_colors.jpg')
    contours, img_contour = find_contour(img, threshold =60, isBlur = True, color = (0, 255, 0), isShow = True)
    feature_list = calc_contour_feature(img, contours, isShow = True)
    print(feature_list)
    img_center = draw_center(img, feature_list, color = (255, 255, 255), radius = 3, isShow = True)
    img_bbox = draw_bbox(img, feature_list, color = (0, 255, 0), width = 1, isShow = True)
    img_bbox2 = draw_bbox2(img, feature_list, color = (0, 255, 0), width = 1, isShow = True)
    img_circle = draw_minSCircle(img, feature_list, color = (0, 255, 0), width = 1, isShow = True)
    return

if __name__ == "__main__":
    main()
