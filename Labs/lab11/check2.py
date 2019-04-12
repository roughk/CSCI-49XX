from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
from PIL import Image
from PIL import ImageOps
import numpy as np
import matplotlib.pyplot as plt



def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

  
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images.shape
			   
len(train_labels)

train_labels

test_images.shape

len(test_labels)

train_images = train_images / 255.0

test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)



'''

a = Image.open('a.jpg')
b = Image.open('b.jpg')
c = Image.open('c.jpg')

a = ImageOps.fit(a,(28,28),method=0,bleed=0.0, centering=(0.5,0.5))
b = ImageOps.fit(b,(28,28),method=0,bleed=0.0, centering=(0.5,0.5))
c = ImageOps.fit(c,(28,28),method=0,bleed=0.0, centering=(0.5,0.5))
a = ImageOps.grayscale(a)
b = ImageOps.grayscale(b)
c = ImageOps.grayscale(c)
a = ImageOps.invert(a)
b = ImageOps.invert(b)
c = ImageOps.invert(c)

a.save("a_edit" +".jpg","JPEG")
b.save("b_edit" +".jpg","JPEG")
c.save("c_edit" +".jpg","JPEG")
a = Image.open('a_edit.jpg')
b = Image.open('b_edit.jpg')
c = Image.open('c_edit.jpg')


np_a = np.array(a)
np_b = np.array(b)
np_c = np.array(c)

my_test_images = []
my_test_images.append(np_a)
my_test_images.append(np_b)
my_test_images.append(np_c)

my_test_images = np.array(my_test_images)

predictions = model.predict(my_test_images)
np.argmax(predictions[0])

for i in range(0,3):
    print("Image #", i)
    print(predictions[i])
    print(np.argmax(predictions[i]))
    print(test_labels[i])
    print(class_names[test_labels[i]])




######
#9000-9014

# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i+9000, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i+9000, predictions, test_labels)
plt.show()
'''