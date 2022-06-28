from nbformat import from_dict
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
path="./data/"
flist=os.listdir(path)
print(flist)
valid_exts=[".jpg",".jpeg",".png"]
for f in flist:
    ext=os.path.splitext(f)[1]
    print(ext)
    if ext.lower() not in valid_exts:
        continue
    img=Image.open(path+f)
    img.show()
    
#cat=Image.open("./data/cat.jpeg")
#cat.show()