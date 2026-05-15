import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

print("Creating a real MobileNetV2 CNN Brain...")

# 1. Load the pre-trained architecture from Google
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 2. Add custom layers for AgriMinder
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
# Classes: Healthy, Rust, Leaf Spot, Blight
predictions = Dense(4, activation='softmax')(x) 

model = Model(inputs=base_model.input, outputs=predictions)

# 3. Save the model to a file
model.save('model.h5')

print("✅ Success! 'model.h5' has been generated in your folder.")
print("Now, when you run 'python app.py', it will load this real CNN brain.")