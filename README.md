# Training with `tf.estimator` and `tf.keras` yields inconsistent results

## Input

![Sample input](./docs/input.png)

## Label

![Sample label](./docs/label.png)

## Overfitting on a single image for 300 epochs with pure `tf.keras`

![Overfitting on a single image with `tf.keras` for 300 epochs produces expected results.](./docs/training-with-pure-keras.png)

## Overfitting on a single image for 300 epochs with the `tf.keras` model converted to `tf.estimator.Estimator`

![Overfitting on a single image with `tf.keras` converted to `tf.estimator.Estimator` produces results that are off completely.](./docs/training-with-estimator-and-conversion.png)

## Overfitting on a single image for 300 epochs with the raw `tf.keras` model in the `tf.estimator` model function 

![Overfitting on a single image with `tf.keras` and `tf.estimator` produces sensible results but are still off.](./docs/training-with-estimator-and-no-conversion.png)
