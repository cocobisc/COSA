from keras.applications import VGG16
from keras import models
from keras.optimizers import Adam, RMSprop
import matplotlib.pyplot as plt
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
classf = open("class.txt", "w")

image_width=224
image_height=224
BATCH_SIZE = 20
EPOCHS=3
TRAIN_DIR = 'dataset/monkeys/training/training'
VAL_DIR = 'dataset/monkeys/validation/validation'
#Load the VGG model
vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(image_width, image_height, 3))

# Freeze the layers except the last 4 layers
for layer in vgg_conv.layers[:-4]:
    layer.trainable = False

# Check the trainable status of the individual layers
for layer in vgg_conv.layers:
    print(layer, layer.trainable)

# Create the model
model = models.Sequential()

# Add the vgg convolutional base model
model.add(vgg_conv)
print(model.input_shape)
# Add new layers
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(
    rescale=1. / 255)

# Change the batchsize according to your system RAM


train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(image_width, image_height),
    batch_size=BATCH_SIZE,
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(image_width, image_height),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False)

for i, name in enumerate(validation_generator.class_indices):
    #string = str(i+1) + " " + name + "\n"
    classf.write(name+"\n")

classf.close()

# Compile the model
model.compile(
    optimizer=RMSprop(lr=1e-4),
    loss='categorical_crossentropy',
    metrics=['acc'])

# Train the model
print(model.input_shape())
history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples//BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    )

# Save the model
model.save('vgg16Monkeys_15.h5')

def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')

    # plt.figure()
    # plt.plot(epochs, loss, 'r.')
    # plt.plot(epochs, val_loss, 'r-')
    # plt.title('Training and validation loss')
    plt.show()

    plt.savefig('acc_vs_epochs.png')
plot_training(history)
