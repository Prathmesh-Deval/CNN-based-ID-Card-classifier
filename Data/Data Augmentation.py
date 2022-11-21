import os
from numpy import expand_dims
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
import cv2

i=0

#Provide path in which you want to perform Data augmentation

path = r'aadhar'
#path = r'pan'

#Traverse throught that directory
img_names = os.listdir(path)
print(img_names)

for img_name in img_names:
    img_path = os.path.join(path, img_name)
    image= cv2.imread(img_path)

    data = img_to_array(image)
    s = expand_dims(data, 0)

    print('okay')
    datagen = ImageDataGenerator(
        #brightness_range=[0.4, 1.5],
        horizontal_flip = True,
        #rescale=1. / 255
        # zoom_range = 0.2
        #shear_range=0.2
        #rotation_range=90
    ) #uncomment and use functions as per needed
    
    #pass our data into data generator

    gen1= datagen.flow(s, batch_size=1)

    for j in range(10):
        batch = gen1.next()
        image=batch[0].astype('uint8')
        
        #save new generated images
        a ='aadhar'+str(i)+'.jpeg'
        i+=1
        path = 'aadhar'
        path = 'pan'
        cv2.imwrite(os.path.join(path, a), image)

