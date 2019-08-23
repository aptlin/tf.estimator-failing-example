# Training with `tf.estimator` and `tf.keras` yields inconsistent results

## Overfitting on a single image with pure `tf.keras`

![Overfitting on a single image with `tf.keras` for 300 epochs produces expected results.](./docs/training-with-pure-keras.png)

## Overfitting on a single image with the `tf.keras` converted to `tf.estimator.Estimator`

![Overfitting on a single image with `tf.keras` converted to `tf.estimator.Estimator` produces results that are off completely.](./docs/training-with-estimator-and-conversion.png)

## Overfitting on a single image with `tf.keras` and `tf.estimator` without conversion

![Overfitting on a single image with `tf.keras` and `tf.estimator` produces sensible results but are still off.](./docs/training-with-pure-keras.png)
