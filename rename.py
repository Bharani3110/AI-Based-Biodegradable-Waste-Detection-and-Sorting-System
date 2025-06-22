import os
os.chdir('pic')
i=1
for file in os.listdir():
    src=file
    dst="pineapple"+"_"+str(i)+".jpg"
    os.rename(src,dst)
    i+=1

