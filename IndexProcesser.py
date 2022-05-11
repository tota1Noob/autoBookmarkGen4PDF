import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import re
from tqdm import tqdm

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from PyPDF2.constants import ImageAttributes as IA
from PyPDF2.constants import PageAttributes as PG
from PyPDF2.constants import Ressources as RES
from PyPDF2.filters import _xobj_to_image
from PyPDF2 import PdfFileReader, PdfFileWriter

from paddleocr import PaddleOCR

class IndexProcesser:
    def __init__(self, filePath, startPageNum, endPageNum, offset):
        """
        IndexProcesser initializer
        args：
            filePath: file path for desired PDF file
            startPageNum: starting page num of index
            endPageNum: ending page num of index
            offset: starting pagenum of the main part
        """
        self.fileName, _ = os.path.splitext(os.path.split(filePath)[1])
        self.output = self.fileName + "_目录标注.pdf"
        self.reader = PdfFileReader(open(filePath, 'rb'))
        self.writer = PdfFileWriter()
        self.indexStart = startPageNum - 1
        self.indexEnd = endPageNum - 1
        self.offset = offset - 1
        self.ocrResults = list()
        self.indexEntries = list()
        self.nestedBookmarks = None

        for page in self.reader.pages:
            self.writer.addPage(page)


    def rockNroll(self):
        self.getOcrResult()
        self.formatIndexEntries()
        self.assignLevels()
        self.generateBookmarks()
        self.saveNewPDF()
    

    def getOcrResult(self):
        print("Initializing PaddleOCR...")
        ocr = PaddleOCR(use_angle_cls=True, lang='ch')
        gotFirstPage = False
        firstPage = -1

        for i in tqdm(range(self.indexStart, self.indexEnd + 1), desc="OCR scanning..."):
            page = self.reader.getPage(i)
            xObject = page[PG.RESOURCES][RES.XOBJECT].getObject()           

            for obj in xObject:
                if xObject[obj][IA.SUBTYPE] == "/Image":
                    extension, byte_stream = _xobj_to_image(xObject[obj])
                    if extension is not None:
                        nparr = np.frombuffer(byte_stream, dtype=np.int8)
                        data = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                        results = ocr.ocr(data, cls=True)
                        if i == self.indexStart:
                            results.pop(0)
                            results.pop(0)
                        
                        while not re.match('\.*\d+', results[-1][1][0]):
                            results.pop()

                        for result in results:
                            indentation = result[0][0][0]
                            content = result[1][0]
                            if re.match('\.*\d+', result[1][0]):
                                content = int(re.findall('\d+', content)[0])
                                if gotFirstPage:
                                    content = content - firstPage + self.offset
                                else:
                                    firstPage = content
                                    content = self.offset
                                    gotFirstPage = True
                            elif content != "目录": #temp fix
                                content = re.sub(r'[….。，]', '', content)
                                if self.ocrResults and not isinstance(self.ocrResults[-1][1], int) and not re.match('第.部分.+', self.ocrResults[-1][1]):
                                    #broken OCR result, breaking a string in half
                                    self.ocrResults[-1][1] += content
                                    continue
                            else:
                                continue

                            self.ocrResults.append([indentation, content, i - self.indexStart])
    

    def formatIndexEntries(self):
        flag = False
        leftmosts = [100000] * (self.indexEnd - self.indexStart + 1)
        for result in tqdm(self.ocrResults, desc="Concatenating OCR results..."):
            if isinstance(result[1], int):
                if result[1] < 0:
                    continue
                self.indexEntries[-1][2] = result[1]
                if flag:
                    self.indexEntries[-2][2] = result[1] - 2
                    flag = False
            elif re.match('第.部分.+', result[1]):
                leftmosts[result[2]] = min(leftmosts[result[2]], result[0])
                flag = True
                sep = re.match('第.部分', result[1]).span()[1]
                self.indexEntries.append([result[0], result[1][:sep] + "  " + result[1][sep:], -1, result[2]])
            else:
                leftmosts[result[2]] = min(leftmosts[result[2]], result[0])
                if re.match('第\d+章', result[1]):
                    sep = re.match('第\d+章', result[1]).span()[1]
                    self.indexEntries.append([result[0], result[1][:sep] + "  " + result[1][sep:], -1, result[2]])
                elif re.match('附录[A-Z]', result[1]):
                    sep = re.match('附录[A-Z]', result[1]).span()[1]
                    self.indexEntries.append([result[0], result[1][:sep] + "  " + result[1][sep:], -1, result[2]])
                else:
                    self.indexEntries.append([result[0], result[1], -1, result[2]])
        
        for entry in tqdm(self.indexEntries, desc="Zeroing left bounds..."):
            entry[0] -= leftmosts[entry[3]]
            entry[0] = int(entry[0])

        self.ocrResults = None
    

    def assignLevels(self):
        print("Clustering...")    
        indentations = np.array([entry[0] for entry in self.indexEntries])
        
        #plt.hist(indentations.reshape(-1))
        #plt.show()

        kmeans = KMeans(n_clusters=3).fit(indentations.reshape(-1, 1))
        levels = kmeans.labels_.tolist()
        centroids = kmeans.cluster_centers_.tolist()
        occur = [(centroids[0], 0), (centroids[1], 1), (centroids[2], 2)]
        occur.sort(reverse=True)
        mapping = dict()
        for i, x in enumerate(occur):
            mapping[x[1]] = i
        for i in range(len(levels)):
            levels[i] = mapping[levels[i]]

        self.nestedBookmarks = [_ for _ in zip(levels, self.indexEntries)] #(level, [leftBound, content, pageNum, indexPage])

    def generateBookmarks(self):
        print(len(self.nestedBookmarks))
        print("Generating book marks...")
        self.writer.addBookmark(title="目录", pagenum=self.indexStart)
        for bookmark in self.nestedBookmarks:
            if bookmark[0] == 2:
                if re.match('第.部分', bookmark[1][1]):
                    greatgrandparent = self.writer.addBookmark(title=bookmark[1][1], pagenum=bookmark[1][2])
                else:
                    grandparent = self.writer.addBookmark(title=bookmark[1][1], pagenum=bookmark[1][2], parent=greatgrandparent)
            elif bookmark[0] == 1:
                parent = self.writer.addBookmark(title=bookmark[1][1], pagenum=bookmark[1][2], parent=grandparent)
            else:
                child = self.writer.addBookmark(title=bookmark[1][1], pagenum=bookmark[1][2], parent=parent)

    def saveNewPDF(self):
        with open(self.output, "wb") as f:
            self.writer.write(f)