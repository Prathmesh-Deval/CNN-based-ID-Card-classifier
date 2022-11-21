import os
from numpy import expand_dims
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
import cv2

i=2669
path = r'Aadhar'
img_names = os.listdir(path)
print(img_names)
p=0
for img_name in img_names:
    if p == 100:
        break
    img_path = os.path.join(path, img_name)
    image= cv2.imread(img_path)

    data = img_to_array(image)
    s = expand_dims(data, 0)

    print('okay')
    datagen = ImageDataGenerator(
        #brightness_range=[0.4, 1.5],
        #horizontal_flip = True,
        rescale=1. / 255
        # zoom_range = 0.2
        #shear_range=0.2
        #rotation_range=90
    )

    it = datagen.flow(s, batch_size=1)

    for j in range(1):
        batch =it.next()
        image=batch[0].astype('uint8')
        a ='aadhar'+str(i)+'.jpeg'
        i+=1
        path = 'Aadhar'
        cv2.imwrite(os.path.join(path, a), image)
        p+=1

