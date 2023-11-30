#!/usr/bin/env python
# -*- coding: utf8 -*-
import sys
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
import codecs
import cv2

XML_EXT = '.xml'
ENCODE_METHOD = 'utf-8'

class PascalVocWriter:

    def __init__(self, foldername, filename, imgSize,databaseSrc='Unknown', localImgPath=None):
        self.foldername = foldername
        self.filename = filename
        self.databaseSrc = databaseSrc
        self.imgSize = imgSize
        self.boxlist = []
        self.localImgPath = localImgPath
        self.verified = False

    def prettify(self, elem):
        """
            Return a pretty-printed XML string for the Element.
        """
        rough_string = ElementTree.tostring(elem, 'utf8')
        root = etree.fromstring(rough_string)
        return etree.tostring(root, pretty_print=True, encoding=ENCODE_METHOD).replace("  ".encode(), "\t".encode())
        # minidom does not support UTF-8
        '''reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="\t", encoding=ENCODE_METHOD)'''

    def genXML(self):
        """
            Return XML root
        """
        # Check conditions
        if self.filename is None or \
                self.foldername is None or \
                self.imgSize is None:
            return None

        top = Element('annotation')
        if self.verified:
            top.set('verified', 'yes')

        folder = SubElement(top, 'folder')
        folder.text = self.foldername

        filename = SubElement(top, 'filename')
        filename.text = self.filename

        if self.localImgPath is not None:
            localImgPath = SubElement(top, 'path')
            localImgPath.text = self.localImgPath

        source = SubElement(top, 'source')
        database = SubElement(source, 'database')
        database.text = self.databaseSrc

        size_part = SubElement(top, 'size')
        width = SubElement(size_part, 'width')
        height = SubElement(size_part, 'height')
        depth = SubElement(size_part, 'depth')
        width.text = str(self.imgSize[1])
        height.text = str(self.imgSize[0])
        if len(self.imgSize) == 3:
            depth.text = str(self.imgSize[2])
        else:
            depth.text = '1'

        segmented = SubElement(top, 'segmented')
        segmented.text = '0'
        return top

    def addBndBox(self, xmin, ymin, xmax, ymax, name, difficult):
        bndbox = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
        bndbox['name'] = name
        bndbox['difficult'] = difficult
        self.boxlist.append(bndbox)

    def appendObjects(self, top):
        for each_object in self.boxlist:
            object_item = SubElement(top, 'object')
            name = SubElement(object_item, 'name')
            try:
                name.text = unicode(each_object['name'])
            except NameError:
                # Py3: NameError: name 'unicode' is not defined
                name.text = each_object['name']
            pose = SubElement(object_item, 'pose')
            pose.text = "Unspecified"
            truncated = SubElement(object_item, 'truncated')
            if int(each_object['ymax']) == int(self.imgSize[0]) or (int(each_object['ymin'])== 1):
                truncated.text = "1" # max == height or min
            elif (int(each_object['xmax'])==int(self.imgSize[1])) or (int(each_object['xmin'])== 1):
                truncated.text = "1" # max == width or min
            else:
                truncated.text = "0"
            difficult = SubElement(object_item, 'difficult')
            difficult.text = str( bool(each_object['difficult']) & 1 )
            bndbox = SubElement(object_item, 'bndbox')
            xmin = SubElement(bndbox, 'xmin')
            xmin.text = str(each_object['xmin'])
            ymin = SubElement(bndbox, 'ymin')
            ymin.text = str(each_object['ymin'])
            xmax = SubElement(bndbox, 'xmax')
            xmax.text = str(each_object['xmax'])
            ymax = SubElement(bndbox, 'ymax')
            ymax.text = str(each_object['ymax'])

    def save(self, targetFile=None):
        root = self.genXML()
        self.appendObjects(root)
        out_file = None
        if targetFile is None:
            out_file = codecs.open(
                self.filename + XML_EXT, 'w', encoding=ENCODE_METHOD)
        else:
            out_file = codecs.open(targetFile, 'w', encoding=ENCODE_METHOD)

        prettifyResult = self.prettify(root)
        out_file.write(prettifyResult.decode('utf8'))
        out_file.close()


class PascalVocReader:
    def __init__(self, filepath):
        # shapes type:
        # [labbel, [(x1,y1), (x2,y2), (x3,y3), (x4,y4)], color, color, difficult]
        self.shapes = []
        self.filepath = filepath
        self.verified = False
        try:
            self.parseXML()
        except:
            pass

    def getShapes(self):
        return self.shapes

    def addShape(self, label, bndbox, difficult):
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        points = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        self.shapes.append((label, points, None, None, difficult))

    def parseXML(self):
        assert self.filepath.endswith(XML_EXT), "Unsupport file format"
        parser = etree.XMLParser(encoding=ENCODE_METHOD)
        xmltree = ElementTree.parse(self.filepath, parser=parser).getroot()
        filename = xmltree.find('filename').text
        try:
            verified = xmltree.attrib['verified']
            if verified == 'yes':
                self.verified = True
        except KeyError:
            self.verified = False

        for object_iter in xmltree.findall('object'):
            bndbox = object_iter.find("bndbox")
            label = object_iter.find('name').text
            # Add chris
            difficult = False
            if object_iter.find('difficult') is not None:
                difficult = bool(int(object_iter.find('difficult').text))
            self.addShape(label, bndbox, difficult)
        return True

def main():
    

    ## Samplle code for writing bbox into xml
    ## 有確認過可以再 LabelImg 打開，且正確位置
    file_path = "./"
    filename = "t1.xml"
    img_fn = "S0925087B,S10,090818_476_121_380800_96800.jpg"
    img = cv2.imread(img_fn, -1)
    img_size = img.shape
    xml_write = PascalVocWriter(foldername= file_path, filename = img_fn, imgSize = img_size)
    #addBndBox(self, xmin, ymin, xmax, ymax, name, difficult)
    bbox_list =[(100, 100, 200, 200, 't1', 0), (300, 300, 400, 400, 't2', 0)]
    for box in bbox_list:
        xml_write.addBndBox(box[0], box[1], box[2], box[3], box[4], box[5])
    xml_write.save("t1.xml")
    return

#################################################    
## read xml and extract class and bboxes
## return class and bbox
def read_xml(xml_file, img_file, isShow = True):
    # sample code for reading xml and draw roi

    xml = PascalVocReader(filepath= xml_file)
    #xml.parseXML()  ## 自動讀取
    shape = xml.getShapes()
    print(shape)
    bbox_list = list()
    for s in shape:
        width = (s[1][1][0] - s[1][0][0])
        height = (s[1][2][1] - s[1][1][1])
        bbox_list.append((s[0], s[1][0][0], s[1][0][1], width, height)) ## class, x1, y1, width, height
        print((s[0], s[1][0][0], s[1][0][1], width, height))
    ## use cv2 to verify the format: lef_top (x1, y1) right_top(x1, )
    if isShow:
        img = cv2.imread(img_file, -1)
        from Draw_Utility import draw_rectangle
        img_display = img.copy()
        print(len(bbox_list))
        for bbox in bbox_list:
            img_display = draw_rectangle(img_display, (bbox[1], bbox[2]), (bbox[1]+bbox[3], bbox[2]+ bbox[4]))
        cv2.imshow(img_fn, img_display)
        cv2.waitKey(0)
    return bbox_list ## (class, x1, y1, width, height)

################################################################
## Writing the bbox into xml file, so LabelImg can read the xml
## bbox_class_list format: (xmin, ymin, xmax, ymax, name)
################################################################
def write_xml(img_filename, bbox_class_list):
    from tien_utility import split_filename_extension
    file_path, filename, ext = split_filename_extension(img_filename)
    # file_path = "./"
    # filename = "t1.xml"
    # img_fn = "S0925087B,S10,090818_476_121_380800_96800.jpg"
    img = cv2.imread(img_filename, -1)
    img_size = img.shape
    xml_write = PascalVocWriter(foldername= file_path, filename = filename, imgSize = img_size)
    #addBndBox(self, xmin, ymin, xmax, ymax, name, difficult)
    for box in bbox_class_list:
        xml_write.addBndBox(box[0], box[1], box[2], box[3], box[4], box[5])
    xml_write.save(file_path+ "/"+ filename + ".xml")
    return

if __name__ =="__main__":
    # bbox_class_list =[(100, 100, 200, 200, 't1', 0), (300, 300, 400, 400, 't2', 0)]
    # img_fn = "./S0925087B,S10,090818_476_121_380800_96800.jpg"
    # write_xml(img_fn, bbox_class_list)
    filename = "./S0925087B,S10,090818_476_121_380800_96800.xml"
    img_fn = "./S0925087B,S10,090818_476_121_380800_96800.jpg"
    class_bbox_list = read_xml(xml_file = filename, img_file = img_fn, isShow = True)

    #main()