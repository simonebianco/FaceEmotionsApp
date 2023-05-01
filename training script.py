import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import itertools
import pandas as pd
import os
from keras.utils import load_img

# PATH
train_path = '.\\FER-2013\\train'
test_path  = '.\\FER-2013\\test'

# FUNCTIONS
def plot_images(train_path):
  emotions = os.listdir(train_path)
  fig, axs = plt.subplots(7, 7, figsize=(18, 10))
  for i, emotion in enumerate(emotions):
      axs[i, 0].text(0.5, 0.5, emotion, ha='center', va='center', fontsize=14)
      axs[i, 0].axis('off')
      emotion_path = os.path.join(train_path, emotion)
      list_files = os.listdir(emotion_path)
      for j in range(6):
          idx = i*6+j
          if(idx < len(axs.flat)):
              image = load_img(os.path.join(emotion_path, list_files[j]))
              axs[i, j+1].imshow(image)
              axs[i, j+1].axis("off")
  plt.suptitle("Emotion Class")
  plt.show()

def load_data(train_path, test_path, IMG_SIZE, BATCH_SIZE):
				train_set = tf.keras.utils.image_dataset_from_directory(
										train_path,
										seed=42,
										validation_split=0.2,
										subset='training',
										image_size=(IMG_SIZE, IMG_SIZE),
										batch_size=BATCH_SIZE,
										labels='inferred',
										shuffle=True,
                    color_mode = "grayscale"
										)

				valid_set = tf.keras.utils.image_dataset_from_directory(
										train_path,
										seed=42,
										validation_split=0.2,
										subset='validation',
										image_size=(IMG_SIZE, IMG_SIZE),
										batch_size=BATCH_SIZE,
										labels='inferred',
										shuffle=True,
										color_mode = "grayscale"
										)

				test_set = tf.keras.utils.image_dataset_from_directory(
										test_path,
										seed=42,
										image_size=(IMG_SIZE, IMG_SIZE),
										batch_size=BATCH_SIZE,
										labels='inferred',
										shuffle=False,
										color_mode = "grayscale"
										)
	
				return train_set, valid_set, test_set

def normalize(img, label):
    img = tf.cast(img, tf.float32)
    label = tf.cast(label, tf.float32)
    return tf.divide(img, 255.), label

def augmentation(image, label):   
    image = tf.image.random_flip_left_right(image)  
    image = tf.image.rot90(image)  
    return image, label

def plot_lrs(train_history, epochs):  
    lrs = 1e-6 * (10 ** (np.arange(epochs) / 20))  
    plt.figure(figsize=(18, 10))   
    plt.semilogx(lrs, train_history.history["val_loss"], color='#f97306', linestyle='dashed', label='Learning Rate')
    plt.legend(loc='best')     
    plt.xlabel("Learning Rate")  
    plt.ylabel("Loss")  
    plt.title("Learning rate vs. loss")  
    plt.show()  
    plt.savefig('lrs.png')

def plot_loss_acc(train_history):  
    epochs = range(len(train_history.history['accuracy']))  
    plt.figure(figsize=(18, 10))  
    plt.subplot(1, 2, 1)  
    plt.plot(epochs, train_history.history['accuracy'], color='#f97306', linestyle='dashed', label='Training accuracy')  
    plt.plot(epochs, train_history.history['val_accuracy'], color='#808080', linestyle='dashed', label='Validation accuracy')  
    plt.legend(loc='best')  
    plt.grid(linewidth=1)  
    plt.title('Training and validation accuracy')  
    plt.subplot(1, 2, 2)  
    plt.plot(epochs, train_history.history['loss'], color='#f97306', linestyle='dashed', label='Training Loss')  
    plt.plot(epochs, train_history.history['val_loss'], color='#808080', linestyle='dashed', label='Validation Loss')  
    plt.legend(loc='best')  
    plt.grid(linewidth=1)  
    plt.title('Training and validation loss')  
    plt.legend()  
    plt.show()   
    plt.savefig('loss_accuracy.png')

def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(18, 10), text_size=7):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] 
    n_classes = cm.shape[0] 
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap='Oranges')
    fig.colorbar(cax)   
    if classes:
      labels = classes
    else:
      labels = np.arange(cm.shape[0])   
    ax.set(title="Confusion Matrix",
           xlabel="Predicted label",
           ylabel="True label",
           xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=labels, 
           yticklabels=labels)
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()
    threshold = (cm.max() + cm.min()) / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
        horizontalalignment="center",
        color="white" if cm[i, j] > threshold else "black",
        size=text_size)

def create_model():
    model = tf.keras.Sequential()
    # Input
    model.add(tf.keras.Input(shape=([48, 48, 1]))),
    # 1 Conv
    model.add(tf.keras.layers.Conv2D(128,(3,3),padding = "same", activation = "relu")),
    model.add(tf.keras.layers.BatchNormalization()),
    model.add(tf.keras.layers.MaxPool2D(2,2)),
    model.add(tf.keras.layers.Dropout(0.25))
    # 2 Conv
    model.add(tf.keras.layers.Conv2D(256,(3,3),padding = "same", activation = "relu")),
    model.add(tf.keras.layers.BatchNormalization()),
    model.add(tf.keras.layers.MaxPool2D(2,2)),
    model.add(tf.keras.layers.Dropout(0.25))
    # 3 Conv
    model.add(tf.keras.layers.Conv2D(512,(3,3),padding = "same", activation = "relu")),
    model.add(tf.keras.layers.BatchNormalization()),
    model.add(tf.keras.layers.MaxPool2D(2,2)),
    model.add(tf.keras.layers.Dropout(0.25))
    # 4 Conv
    model.add(tf.keras.layers.Conv2D(512,(3,3),padding = "same", activation = "relu")),
    model.add(tf.keras.layers.BatchNormalization()),
    model.add(tf.keras.layers.MaxPool2D(2,2)),
    model.add(tf.keras.layers.Dropout(0.25))
    # Flattend
    model.add(tf.keras.layers.Flatten())
    # 1 Dense
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.25))
    # 2 Dense
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.25))
    # Output
    model.add(tf.keras.layers.Dense(7, activation='softmax'))

    return model
      
## CALLBACKS
class AccuracyCallback(tf.keras.callbacks.Callback):  
    def on_epoch_end(self, epoch, logs={}):  
        if logs.get('accuracy') >= 0.95:  
            print("\nReached 95% accuracy so cancelling training")  
            self.model.stop_training = True  
  
accuracy_call = AccuracyCallback()  
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1) 
checkpoint = tf.keras.callbacks.ModelCheckpoint('model.h5', save_best_only=True)  
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-6 * 10 ** (epoch / 20))
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                factor=0.2,
                                                patience=3,
                                                verbose=1,
                                                min_delta=0.0001)

callbacks_list = [accuracy_call, reduce_lr]

# PLOT IMAGES
plot_images(train_path)

# PARAMETERS
IMG_SIZE = 48
BATCH_SIZE = 64
AUTOTUNE = tf.data.AUTOTUNE

# LOAD DATA
train_set, valid_set, test_set = load_data(train_path, test_path, IMG_SIZE, BATCH_SIZE)

# NORMALIZE  DATA
train_set = train_set.map(normalize, num_parallel_calls=AUTOTUNE)
valid_set = valid_set.map(normalize, num_parallel_calls=AUTOTUNE)
test_set = test_set.map(normalize, num_parallel_calls=AUTOTUNE)

# AUGMENT DATA
#train_set = train_set.map(augmentation, num_parallel_calls=AUTOTUNE)
#valid_set = valid_set.map(augmentation, num_parallel_calls=AUTOTUNE)

# CACHE DATE
train_set = train_set.cache().prefetch(AUTOTUNE) 
valid_set = valid_set.cache().prefetch(AUTOTUNE) 

# MODEL CREATE 
model = create_model()

# COMPILE MODEL
opt = tf.keras.optimizers.Adam(learning_rate = 0.001)
model.compile(optimizer = opt,
              loss = "sparse_categorical_crossentropy",
              metrics = ["accuracy"]) 

# FIT MODEL
tf.keras.backend.clear_session()  
EPOCHS = 50
history = model.fit(train_set, 
                    validation_data=valid_set, 
                    epochs=EPOCHS, 
                    verbose=1,
                    callbacks=callbacks_list) 

# SAVE JSON MODEL
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# SAVE WEIGHTS 
model.save_weights("weight.h5")

# SAVE HISTORY 
hist_df = pd.DataFrame(history.history) 
hist_csv_file = 'history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

# PRED LABELS
test_prediction = model.predict(test_set)
test_prediction_2 = test_prediction.argmax(axis=1)

# TRUE LABELS
test_label = []
for features, label in test_set:
    test_label.append(label.numpy())

test_label_2 = []
for i in test_label:
    for j in i:
        test_label_2.append(j)

# CONFUSION MATRIX
classes = ['Angry', 'Disgust', 'Fear', 'Happy','Sad', 'Surprise', 'Neutral']
make_confusion_matrix(test_label_2, test_prediction_2, classes=classes)       

# CLASSIFICATION REPORT
cr = classification_report(test_label_2, test_prediction_2, target_names=classes, output_dict=True)
df = pd.DataFrame(cr).transpose()
df = df.apply(lambda x: x.round(2))
df.to_csv('classification_report.csv')








