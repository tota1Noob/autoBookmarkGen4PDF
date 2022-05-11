import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from paddleocr import PaddleOCR,draw_ocr
# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# You can set the parameter `lang` as `ch`, `en`, `fr`, `german`, `korean`, `japan`
# to switch the language model in order.
ocr = PaddleOCR(use_angle_cls=True, lang='ch') # need to run only once to download and load model into memory
img_path = 'Im6.png'
result = ocr.ocr(img_path, cls=True)
#for line in result:
#    print(line)

# Each line consists of a 4 * 2 list and a tuple, 
# containing coordinates of a bounding box and ocr result with confidence, respectively.

# draw result
from PIL import Image
image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txts, scores, font_path='./fonts/simfang.ttf')
im_show = Image.fromarray(im_show)
im_show.save('result.jpg')