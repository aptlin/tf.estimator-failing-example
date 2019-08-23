import tensorflow as tf
import numpy as np
from segmentation_models import FPN
import matplotlib.pyplot as plt

model = FPN(
    backbone_name="mobilenetv2",
    input_shape=(None, None, 3),
    classes=7,
    activation="sigmoid",
    weights=None,
    encoder_weights="imagenet",
    encoder_features="default",
    pyramid_block_filters=256,
    pyramid_use_batchnorm=True,
    pyramid_aggregation="concat",
    pyramid_dropout=None,
)
image = np.load("./data/image.npy")
label = np.load("./data/label.npy")

model.compile(
    loss=lambda labels, predictions: tf.keras.losses.binary_crossentropy(
        labels[:, :, :, 1:], predictions
    ),
    optimizer="adam",
    metrics=["accuracy"],
)

model.fit(
    [image],
    [label],
    batch_size=1,
    epochs=300,
    verbose=2,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
            filepath="./dist/keras/psenet", save_weights_only=True
        )
    ],
)
kernels = model.predict(image)

plt.figure()
plt.imshow(np.squeeze(kernels[:, :, :, 0]))
plt.savefig("./docs/training-with-pure-keras.png")
