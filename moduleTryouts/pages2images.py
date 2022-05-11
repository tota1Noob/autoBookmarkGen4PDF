import PyPDF2
from PyPDF2.constants import ImageAttributes as IA
from PyPDF2.constants import PageAttributes as PG
from PyPDF2.constants import Ressources as RES
from PyPDF2.filters import _xobj_to_image
from PyPDF2 import PdfFileReader, PdfFileWriter

import cv2
import numpy as np


if __name__ == '__main__':
    path = "..\Python学习手册下册.pdf"
    reader = PdfFileReader(open(path, 'rb'))
    writer = PdfFileWriter()

    page = reader.getPage(10)
    xObject = page[PG.RESOURCES][RES.XOBJECT].getObject()
    """
    for obj in xObject:
        if xObject[obj][IA.SUBTYPE] == "/Image":
            extension, byte_stream = _xobj_to_image(xObject[obj])
            if extension is not None:
                filename = obj[1:] + ".png"
                with open(filename, "wb") as img:
                    img.write(byte_stream)
    """

    for obj in xObject:
        if xObject[obj][IA.SUBTYPE] == "/Image":
            extension, byte_stream = _xobj_to_image(xObject[obj])
            if extension is not None:
                nparr = np.frombuffer(byte_stream, dtype=np.uint8)
                data = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                import os
                os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

                from paddleocr import PaddleOCR,draw_ocr
                # Paddleocr supports Chinese, English, French, German, Korean and Japanese.
                # You can set the parameter `lang` as `ch`, `en`, `fr`, `german`, `korean`, `japan`
                # to switch the language model in order.
                ocr = PaddleOCR(use_angle_cls=True, lang='ch') # need to run only once to download and load model into memory
                img_path = 'Im6.png'
                result = ocr.ocr(data, cls=True)
                #for line in result:
                #    print(line)

                # Each line consists of a 4 * 2 list and a tuple, 
                # containing coordinates of a bounding box and ocr result with confidence, respectively.

                # draw result
                from PIL import Image
                image = Image.fromarray(data).convert('RGB')
                boxes = [line[0] for line in result]
                txts = [line[1][0] for line in result]
                scores = [line[1][1] for line in result]
                im_show = draw_ocr(image, boxes, txts, scores, font_path='./fonts/simfang.ttf')
                im_show = Image.fromarray(im_show)
                im_show.save('result.jpg')