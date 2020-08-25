import math, json, os, sys
import matplotlib.pyplot as plt
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, MaxPool2D, Flatten, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications import ResNet50
classf = open("class.txt", "w")
TRAIN_DIR = 'PCB/train'
VALID_DIR = 'PCB/test'
image_width=64
image_height=64
BATCH_SIZE = 15
SIZE=(image_width, image_height)

def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')

    plt.show()

if __name__ == "__main__":
    num_train_samples = sum([len(files) for r, d, files in os.walk(TRAIN_DIR)])
    num_valid_samples = sum([len(files) for r, d, files in os.walk(VALID_DIR)])

    num_train_steps = math.floor(num_train_samples//BATCH_SIZE)
    num_valid_steps = math.floor(num_valid_samples//BATCH_SIZE)

    train_datagen = ImageDataGenerator()
    val_datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)

    train_generator = train_datagen.flow_from_directory(TRAIN_DIR, target_size=SIZE, class_mode='categorical', shuffle=True, batch_size=BATCH_SIZE)
    val_generator = val_datagen.flow_from_directory(VALID_DIR, target_size=SIZE, class_mode='categorical', shuffle=True, batch_size=BATCH_SIZE)

    model = ResNet50(weights='imagenet', include_top=False, input_shape=(image_width, image_height, 3))

    classes = list(iter(train_generator.class_indices))
    for layer in model.layers[:-16]:
        layer.trainable = False

    last = model.output

    last = GlobalAveragePooling2D()(last)
    last = Dropout(0.5)(last)

    x = Dense(len(classes), activation="softmax")(last)

    finetuned_model = Model(model.input, x)
    finetuned_model.summary()
    for i, name in enumerate(val_generator.class_indices):
        # string = str(i+1) + " " + name + "\n"
        classf.write(name + "\n")
    finetuned_model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    for c in train_generator.class_indices:
        classes[train_generator.class_indices[c]] = c
    finetuned_model.classes = classes

    history=finetuned_model.fit_generator(train_generator, steps_per_epoch=num_train_steps, epochs=10, validation_data=val_generator, validation_steps=num_valid_steps)
    finetuned_model.save('QFN_Dicing_Resnet50.h5')
    plot_training(history)