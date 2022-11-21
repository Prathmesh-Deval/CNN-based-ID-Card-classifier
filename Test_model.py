from keras.models import load_model
import cv2
import numpy as np

labels_dict={0:'Aadhar Card',1:'Pan Card'}


model = load_model('Card_Detector.model')
img_path ='me_aadhar.jpeg'
img= cv2.imread(img_path)


window_name = 'image'
cv2.imshow(window_name, img)


gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
resized = cv2.resize(gray, (100, 100))
normalized = resized / 255.0
reshaped = np.reshape(normalized, (1, 100, 100, 1))
result=model.predict(reshaped)
print(result)
label = np.argmax(result, axis=1)[0]
print(labels_dict[label])

cv2.waitKey(0)
cv2.destroyAllWindows()
