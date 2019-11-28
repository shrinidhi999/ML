from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten

classifier = Sequential()

classifier.add(Conv2D(32,(3,3), activation = 'relu', input_shape=(64,64,3)))
classifier.add(MaxPool2D(pool_size=(2,2)))
classifier.add(Flatten())

classifier.add(Dense(128, activation='relu'))
classifier.add(Dense(1, activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory(
        r'D:\DL\Deep-Learning-AZ\Volume 1 - Supervised Deep Learning\Part 2 - Convolutional Neural Networks (CNN)\Section 8 - Building a CNN\P16-Convolutional-Neural-Networks\Convolutional_Neural_Networks\dataset\training_set',
        target_size=(64, 64),
        batch_size=128,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        r'D:\DL\Deep-Learning-AZ\Volume 1 - Supervised Deep Learning\Part 2 - Convolutional Neural Networks (CNN)\Section 8 - Building a CNN\P16-Convolutional-Neural-Networks\Convolutional_Neural_Networks\dataset\test_set',
        target_size=(64, 64),
        batch_size=128,
        class_mode='binary')

import multiprocessing

classifier.fit_generator(train_set,
        steps_per_epoch=8000,
        epochs=1,
        workers=multiprocessing.cpu_count(),
        validation_data=test_set,
        validation_steps=2000)



import numpy as np
from keras.preprocessing import image

test_image = image.load_img('D:\DL\Deep-Learning-AZ\Volume 1 - Supervised Deep Learning\Part 2 - Convolutional Neural Networks (CNN)\Section 8 - Building a CNN\P16-Convolutional-Neural-Networks\Convolutional_Neural_Networks\dataset\single_prediction\cat_or_dog_1.jpg', target_size = (64, 64))

test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
train_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
    
    

test_image = image.load_img('D:\DL\Deep-Learning-AZ\Volume 1 - Supervised Deep Learning\Part 2 - Convolutional Neural Networks (CNN)\Section 8 - Building a CNN\P16-Convolutional-Neural-Networks\Convolutional_Neural_Networks\dataset\single_prediction\cat_or_dog_2.jpg', target_size = (64, 64))

test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
train_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'    