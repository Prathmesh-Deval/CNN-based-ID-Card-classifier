import cv2,os
import numpy as np
from keras.utils import np_utils
from sklearn.utils import shuffle


import warnings
warnings.filterwarnings("ignore")




data_path= r'Data1'

categories = ["Aadhar", "Pan"]

labels=[i for i in range(len(categories))]
label_dict=dict(zip(categories,labels))



img_size=100
data=[]
target=[]
for category in categories:
	folder_path=os.path.join(data_path,category)
	img_names=os.listdir(folder_path)
	for img_name in img_names:
		img_path=os.path.join(folder_path,img_name)
		print(img_path)
		img=cv2.imread(img_path)

		gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		resized=cv2.resize(gray,(img_size,img_size))

		data.append(resized)
		target.append(label_dict[category])


data=np.array(data)/255.0
data=np.reshape(data,(-1,img_size,img_size,1))

target=np.array(target)

data, target = shuffle(data, target)
target=np_utils.to_categorical(target)

np.save('data',data)
np.save('target',target)
