
import PIL.Image as Image
from PIL import ImageFilter
from PIL import Image
import cv2
import os
 

path=r"C:\Users\w1360\Desktop\Labels"
save=r"C:\Users\w1360\Desktop\RGB_filter"
files = os.listdir(path)
os.chdir(path)

for file in files:

   img = Image.open(file)
 
   img_array = img.load()
   
   width, height = img.size
   
   for x in range(0,width):
     for y in range(0,height):
        rgb = img_array[x,y]
        r = rgb[0]
        g = rgb[1]
        b = rgb[2]
        if r==255 and g==0 and b==0:
           img_array[x, y] =(255,0,0);
        else:
           img_array[x, y] =(0,0,0)
        
            
   
   img_name=file.split('.')[0]
   img.save(os.path.join(save,img_name+'.png'))
 