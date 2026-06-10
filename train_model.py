import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import os

# 1. Image Settings
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
# Yahan check karein ke aapka folder 'dataset/train' hai ya sirf 'dataset'
DATASET_PATH = 'dataset/train' 

# 2. Smart Data Generator (Auto-Split enabled)
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    validation_split=0.2 # 20% data validation ke liye khud nikal lega
)

# Training Data
train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training' # 80% data
)

# Validation Data
val_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation' # 20% data
)

# 3. Build Model (MobileNetV2)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False 

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.2)(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# 4. Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 5. Start Training
print(f"Classes detected: {train_generator.num_classes}")
print("Starting Training... This will take time based on your Laptop Speed.")

model.fit(
    train_generator,
    epochs=5, # FYP demo ke liye 5-10 kafi hain
    validation_data=val_generator
)

# 6. Save
model.save('model.h5')
print("✅ Success! 'model.h5' generated.")