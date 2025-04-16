import os
import sys
import argparse
import collections
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, project_root)

from training.data import SegmentationDataset
from training.model import BaseModel
from training.common import SegmentationLoss
from training.common.callbacks import SegmentationLog

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', type=str, required=True)
parser.add_argument('--val_image_dir', type=str, required=True)
parser.add_argument('--val_label_dir', type=str, required=True)
parser.add_argument('--use_small_train', action='store_true')
args = parser.parse_args()

if args.use_small_train:
    train_dir = os.path.join(args.train_dir + "_500")
    print("✅ Using only 500 small training set!")
else:
    train_dir = args.train_dir

val_image_dir = args.val_image_dir
val_label_dir = args.val_label_dir

# GPU
physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

num_epochs = 10
batch_size = 12
num_classes = 1

train_dataset = SegmentationDataset(image_dir=train_dir + "/image", label_dir=train_dir + "/label", batch_size=batch_size, resize=(224, 224))
val_dataset = SegmentationDataset(image_dir=val_image_dir, label_dir=val_label_dir, batch_size=20, resize=(224, 224))

model = BaseModel(num_classes, (None, 224, 224, 3))
model.make_model()
model.summary()

metrics_to_print = collections.OrderedDict([
    ("loss", "loss"), ("val_loss", "val_loss"),
    ("accuracy", "accuracy"), ("val_accuracy", "val_accuracy")
])

callbacks = [
    ModelCheckpoint(filepath="training/experiments/logs/ckpts/weights-epoch{epoch:02d}.keras", save_freq='epoch'),
    TensorBoard(log_dir="training/experiments/logs/tensorboard", histogram_freq=0, write_graph=True),
    SegmentationLog(metrics_to_print, txt_log_path="training/experiments/logs", val=True, val_data=val_dataset, num_epochs=num_epochs)
]

model.model.compile(optimizer='adam', loss=SegmentationLoss(), metrics=["accuracy"])
model_history = model.model.fit(train_dataset, epochs=num_epochs, callbacks=callbacks, validation_data=val_dataset, verbose=1)

os.makedirs("training/experiments/logs/weights", exist_ok=True)

model.model.save("training/experiments/logs/weights/sidewalk-detect.keras")

# Plot training metrics directly
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(model_history.history['loss'], label='Train Loss')
plt.plot(model_history.history['val_loss'], label='Val Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(model_history.history['accuracy'], label='Train Accuracy')
if 'val_accuracy' in model_history.history:
    plt.plot(model_history.history['val_accuracy'], label='Val Accuracy')
elif 'val_acc' in model_history.history:
    plt.plot(model_history.history['val_acc'], label='Val Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


