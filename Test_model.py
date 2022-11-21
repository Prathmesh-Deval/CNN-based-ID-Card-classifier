from keras.models import load_model
import cv2
import numpy as np

#Lables to provide to maximum value from predicted outcome
labels_dict={0:'Aadhar Card',1:'Pan Card'}

#load our trained model
model = load_model('Card_Detector.model')

#provide path of the image that u want to predict(aadhar card/pan card)
img_path ='me_aadhar.jpeg'
img= cv2.imread(img_path)

#show input image
window_name = 'image'
cv2.imshow(window_name, img)

#process the image in same way that we used for data preprocessing for training our model
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
resized = cv2.resize(gray, (100, 100))
normalized = resized / 255.0
reshaped = np.reshape(normalized, (1, 100, 100, 1))

#Predict image with our module
result=model.predict(reshaped)

#take max value and provide lables from our dictionary
label = np.argmax(result, axis=1)[0]

#show output
print(labels_dict[label])

cv2.waitKey(0)
cv2.destroyAllWindows()
