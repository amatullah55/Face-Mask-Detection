import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

INIT_LR = 1e-4
EPOCHS = 25
BS = 32

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    "dataset",
    target_size=(224, 224),
    batch_size=BS,
    class_mode="categorical",
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    "dataset",
    target_size=(224, 224),
    batch_size=BS,
    class_mode="categorical",
    subset='validation'
)

# Load base model
baseModel = MobileNetV2(weights="imagenet", include_top=False,
                        input_shape=(224, 224, 3))

# Build head
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten()(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

# Freeze most layers, unfreeze last 20 layers
for layer in baseModel.layers[:-20]:
    layer.trainable = False

for layer in baseModel.layers[-20:]:
    layer.trainable = True

# Compile
opt = Adam(learning_rate=INIT_LR)
model.compile(loss="categorical_crossentropy",
              optimizer=opt,
              metrics=["accuracy"])

# Train
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# Save
model.save("mask_detector.h5")

print("✅ Model trained successfully")