import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ---------------- Load dataset ---------------- #

X = np.load("X_spectrogram.npy")
y = np.load("y_labels.npy")

print("Dataset shape:", X.shape)

# Normalize spectrogram values
X = (X - X.min()) / (X.max() - X.min())

# Reshape for CNN
X = X.reshape(X.shape[0], 64, 128, 1)

# Convert labels to categorical
y_cat = to_categorical(y)

# ---------------- Train/Test split ---------------- #

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_cat,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------- Handle class imbalance ---------------- #

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y),
    y=y
)

class_weights = dict(enumerate(class_weights))

print("Class Weights:", class_weights)

# ---------------- CNN Architecture ---------------- #

model = Sequential([

    Input(shape=(64,128,1)),

    Conv2D(32,(3,3),activation='relu',padding="same"),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64,(3,3),activation='relu',padding="same"),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128,(3,3),activation='relu',padding="same"),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(256,(3,3),activation='relu',padding="same"),
    BatchNormalization(),

    GlobalAveragePooling2D(),

    Dense(128,activation='relu'),
    Dropout(0.5),

    Dense(2,activation='softmax')
])

# ---------------- Compile Model ---------------- #

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name="auc")
    ]
)

model.summary()

# ---------------- Callbacks ---------------- #

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=6,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=3,
    min_lr=1e-6
)

checkpoint = ModelCheckpoint(
    "cnn_best_model.keras",
    monitor="val_auc",
    save_best_only=True
)

# ---------------- Train Model ---------------- #

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test,y_test),
    epochs=50,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[early_stop,reduce_lr,checkpoint]
)

# ---------------- Save Final Model ---------------- #

model.save("cnn_murmur_model.keras")

print("CNN model training complete.")