import cv2,os
import numpy as np
from keras.utils import np_utils
from sklearn.utils import shuffle

#provide out data path
data_path= r'Data'

categories = ["Aadhar", "Pan"]

labels=[i for i in range(len(categories))]
label_dict=dict(zip(categories,labels))

img_size=100

data=[]
target=[]

#traverse through aadhar and pan data folders
for category in categories:
	folder_path=os.path.join(data_path,category)
	img_names=os.listdir(folder_path)
	for img_name in img_names:
		img_path=os.path.join(folder_path,img_name)
		print(img_path)
		#convert image to gray scale and resize
		img=cv2.imread(img_path)

		gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		resized=cv2.resize(gray,(img_size,img_size))

		data.append(resized)
		target.append(label_dict[category])

#normalize our data
data=np.array(data)/255.0
data=np.reshape(data,(-1,img_size,img_size,1))

target=np.array(target)

data, target = shuffle(data, target)
target=np_utils.to_categorical(target)

#save our data in below format to load for future use
np.save('data',data)
np.save('target',target)
