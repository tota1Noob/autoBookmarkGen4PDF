from PyPDF2 import PdfFileReader, PdfFileWriter
import os

if __name__ == '__main__':
    path = "..\Python学习手册下册.pdf"
    fileName, extension = os.path.splitext(os.path.split(path)[1])
    reader = PdfFileReader(open(path, 'rb'))
    writer = PdfFileWriter()

    for page in reader.pages:
        writer.addPage(page)
    
    parent = writer.addBookmark(title="概览OOP", pagenum=24)
    child = writer.addBookmark(title="属性继承搜索", pagenum=24, parent=parent)

    with open(fileName + "_目录标注.pdf", "wb") as f:
        writer.write(f)