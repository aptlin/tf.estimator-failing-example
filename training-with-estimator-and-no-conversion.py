import numpy as np
import tensorflow as tf
import segmentation_models as sm
import matplotlib.pyplot as plt

sm.set_framework("tf.keras")


def build_model():
    images = tf.keras.Input(shape=[None, None, 3], name="image", dtype=tf.float32)
    model = sm.FPN(
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
    )(images)

    return tf.keras.Model(inputs={"image": images}, outputs=model)


def input_fn(image, label=None):
    return tf.estimator.inputs.numpy_input_fn(
        {"image": image}, label, num_epochs=None, batch_size=1, shuffle=False
    )


model = build_model()
model.compile(
    loss=lambda labels, predictions: tf.keras.losses.binary_crossentropy(
        labels[:, :, :, 1:], predictions
    ),
    optimizer="adam",
    metrics=["accuracy"],
)
estimator = tf.keras.estimator.model_to_estimator(
    keras_model=model, model_dir="dist/estimator/no-conversion"
)

image = np.load("./data/image.npy")
label = np.load("./data/label.npy")

estimator.train(input_fn=input_fn(image, label), steps=300)
preds = next(estimator.predict(input_fn=input_fn(image)))
kernels = preds["model"]

plt.figure()
plt.imshow(kernels[:, :, 0])
plt.savefig("./docs/training-with-estimator-and-no-conversion.png")

