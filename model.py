import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Load the MobileNetV2 model with pretrained weights, excluding the top classification layer
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Freeze the base model layers
base_model.trainable = False

# Add custom layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)  # Add Dropout for regularization
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

# Define the model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Learning Rate Scheduler
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

# Early Stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Define initial batch size and create ImageDataGenerators
initial_batch_size = 32
batch_size_increase_epoch = 5
max_batch_size = 64

# Data augmentation for training
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Data augmentation for validation (no augmentation, only rescaling)
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

def create_generators(batch_size):
    train_generator = train_datagen.flow_from_directory(
        directory=data_dir,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        directory='/kaggle/input/nepali-food-images/dataset/test',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_generator, val_generator

# Train the model with gradual batch size increase
for epoch in range(10):  # Number of epochs
    if epoch % batch_size_increase_epoch == 0 and initial_batch_size < max_batch_size:
        initial_batch_size = min(max_batch_size, initial_batch_size * 2)  # Double the batch size
        print(f"Increasing batch size to {initial_batch_size}")
    
    train_generator, val_generator = create_generators(initial_batch_size)
    
    history = model.fit(
        train_generator,
        epochs=10,
        validation_data=val_generator,
        callbacks=[lr_scheduler, early_stopping]  # Use the learning rate scheduler and early stopping
    )

# Save the model after training
model.save('/kaggle/working/mobilenetv2_food_model.h5')

Found 1548 images belonging to 9 classes.
Found 395 images belonging to 9 classes.
25/25 ━━━━━━━━━━━━━━━━━━━━ 45s 1s/step - accuracy: 0.4234 - loss: 9.4223 - val_accuracy: 0.7089 - val_loss: 7.7973 - learning_rate: 0.0010
25/25 ━━━━━━━━━━━━━━━━━━━━ 26s 801ms/step - accuracy: 0.7357 - loss: 7.5516 - val_accuracy: 0.7165 - val_loss: 6.9755 - learning_rate: 0.0010
25/25 ━━━━━━━━━━━━━━━━━━━━ 25s 822ms/step - accuracy: 0.7828 - loss: 6.6675 - val_accuracy: 0.8076 - val_loss: 6.0123 - learning_rate: 0.0010
25/25 ━━━━━━━━━━━━━━━━━━━━ 25s 801ms/step - accuracy: 0.8017 - loss: 5.7765 - val_accuracy: 0.7924 - val_loss: 5.2846 - learning_rate: 0.0010
25/25 ━━━━━━━━━━━━━━━━━━━━ 25s 767ms/step - accuracy: 0.8182 - loss: 5.0963 - val_accuracy: 0.7620 - val_loss: 4.7128 - learning_rate: 0.0010
25/25 ━━━━━━━━━━━━━━━━━━━━ 25s 765ms/step - accuracy: 0.8206 - loss: 4.4633 - val_accuracy: 0.7772 - val_loss: 4.1920 - learning_rate: 0.0010
25/25 ━━━━━━━━━━━━━━━━━━━━ 26s 800ms/step - accuracy: 0.8317 - loss: 3.9238 - val_accuracy: 0.8076 - val_loss: 3.7118 - learning_rate: 0.0010
25/25 ━━━━━━━━━━━━━━━━━━━━ 25s 768ms/step - accuracy: 0.8658 - loss: 3.4606 - val_accuracy: 0.8127 - val_loss: 3.3233 - learning_rate: 0.0010
25/25 ━━━━━━━━━━━━━━━━━━━━ 26s 779ms/step - accuracy: 0.8555 - loss: 3.0625 - val_accuracy: 0.8025 - val_loss: 2.9889 - learning_rate: 0.0010
25/25 ━━━━━━━━━━━━━━━━━━━━ 26s 768ms/step - accuracy: 0.8458 - loss: 2.7934 - val_accuracy: 0.8051 - val_loss: 2.6915 - learning_rate: 0.0010
