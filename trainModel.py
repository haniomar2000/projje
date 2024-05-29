# In[1]:
# Import required libraries and models
import os
import matplotlib.pyplot as plt
import math
from tensorflow.keras.models import Sequential
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout, SpatialDropout2D
from tensorflow.keras.losses import sparse_categorical_crossentropy, binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from tensorflow.keras.callbacks import History
from keras import callbacks
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

# In[2]:
# Specify the input training datsaet directory
dir_example = "./Data"
classes = os.listdir(dir_example)
print("Total Number of Classes Found in the Directory is: ",classes)

dir_example = "./Data/Train"
train_classes = os.listdir(dir_example)
print("Total Number of Classes Found in the Train Directory is: ", train_classes)

# In[3]:
train = './Data/Train'
test = './Data/Test'
# Create an Image Generator for the CNN model
train_generator = ImageDataGenerator(rescale = 1/255)
# Create a training generator for the data with input size of 300x300 pixels
train_generator = train_generator.flow_from_directory(train, target_size = (300,300), batch_size = 32, class_mode = 'sparse')
# Extract labels from training data
labels = (train_generator.class_indices)
print("Labels of the Train Generator are: ", labels)
labels = dict((v,k) for k,v in labels.items())
print("Train Test Dictionary Labels are: Dictionary Labels are: ", labels)

# In[4]:
# Scale all images colors into GRAY
test_generator = ImageDataGenerator(rescale = 1./255)
test_generator = test_generator.flow_from_directory(test, target_size = (300,300), batch_size = 32, class_mode = 'sparse')
test_labels = (test_generator.class_indices)
print("Labels of the Test Generator are: ", test_labels)
test_labels = dict((v,k) for k,v in test_labels.items())
print("", test_labels)

for image_batch, label_batch in train_generator:
  break
print(image_batch.shape)
print(label_batch.shape)

print(train_generator.class_indices)
Labels = '\n'.join(sorted(train_generator.class_indices.keys()))
with open('Labels.txt', 'w') as file:
  file.write(Labels)

# In[5]:
# Create the Training CNN Model architecture
model=Sequential()
#Convolution blocks
model.add(Conv2D(32, kernel_size = (3,3), padding='same',input_shape=(300,300,3),activation='relu'))
model.add(MaxPooling2D(pool_size=2)) 
model.add(Conv2D(64, kernel_size = (3,3), padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=2)) 
model.add(Conv2D(32, kernel_size = (3,3), padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=2)) 
#Classification layers
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(6,activation='softmax'))
model.summary()

# In[6]:
earlystopping = callbacks.EarlyStopping(monitor ="loss", 
                                        mode ="min", patience = 10, 
                                        restore_best_weights = True)

model.compile(optimizer = 'Adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
history = model.fit_generator(train_generator, validation_data=test_generator, epochs=50, steps_per_epoch=100//32, callbacks=[earlystopping])

model.save('finalModel.h5')
# Load training model
model = load_model('finalModel.h5')

# In[7]:
# Plot accuracy curve
model = load_model('finalModel.h5')

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot loss curve
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# In[8]:
from keras.preprocessing import image
import numpy as np
test_img = './Data/Train/Sick/X3.jpg'
img = image.load_img(test_img, target_size = (300,300))
# Reshape the image to array dimension
img = image.img_to_array(img, dtype=np.uint8)
# Scale the image into GRAY
img = np.array(img)/255.0
# Apply the trained model to predict the label/category
prediction = model.predict(img[np.newaxis, ...])
#print("Predicted shape",p.shape)
print("Probability:",np.max(prediction[0], axis=-1))
predicted_class = labels[np.argmax(prediction[0], axis=-1)]
# Print the label and probability
print("Classified:",predicted_class,'\n')
plt.axis('off')
plt.imshow(img.squeeze())
plt.title("Loaded Image")

# In[9]:
from keras.preprocessing import image
import numpy as np
import glob

# Get the true labels
true_labels = []
files = glob.glob("./Data/Test/*/*")
for file in files:
    if "Healthy" in file:
        true_labels.append(0)
    elif "Sick" in file:
        true_labels.append(1)
    # add more elif conditions based on your class names

# Generate predictions
pred_labels = []
for file in files:
    img = image.load_img(file, target_size = (300,300))
    img = image.img_to_array(img, dtype=np.uint8)
    img = np.array(img)/255.0
    prediction = model.predict(img[np.newaxis, ...])
    pred_class = np.argmax(prediction[0], axis=-1)
    pred_labels.append(pred_class)

# Create a confusion matrix
cm = confusion_matrix(true_labels, pred_labels)

# Plot the confusion matrix
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()
