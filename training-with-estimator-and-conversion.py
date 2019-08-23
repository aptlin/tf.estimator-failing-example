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


def model_fn(features, labels, mode, params):
    image = features["image"]
    model = build_model()
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = model(image, training=False)
        predictions = {"kernels": predictions}
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions,
            export_outputs={"detect": tf.estimator.export.PredictOutput(predictions)},
        )
    elif mode == tf.estimator.ModeKeys.TRAIN:
        masks = labels[:, :, :, 0]
        ground_truth = labels[:, :, :, 1:]
        predictions = model(image, training=True)
        loss = tf.math.reduce_mean(
            tf.keras.losses.binary_crossentropy(ground_truth, predictions)
        )
        optimizer = tf.compat.v1.train.AdamOptimizer()
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            train_op=optimizer.minimize(
                loss, tf.compat.v1.train.get_or_create_global_step()
            ),
        )
    elif mode == tf.estimator.ModeKeys.EVAL:
        predictions = model(features["image"], training=False)
        loss = tf.math.reduce_mean(
            tf.keras.losses.binary_crossentropy(ground_truth, predictions)
        )
        return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.EVAL, loss=loss)
    else:
        raise ValueError("The mode {} is not supported, aborting.".format(mode))


estimator = tf.estimator.Estimator(
    model_fn=model_fn, model_dir="dist/estimator/conversion/"
)

image = np.load("./data/image.npy")
label = np.load("./data/label.npy")
estimator.train(input_fn=input_fn(image, label), steps=300)
preds = next(estimator.predict(input_fn=input_fn(image)))
kernels = preds["kernels"]

plt.figure()
plt.imshow(kernels[:, :, 0])
plt.savefig("./docs/training-with-estimator-and-conversion.png")

