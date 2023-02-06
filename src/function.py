import tensorflow as tf

from src.metrics import calc_coord_accuracy


@tf.function
def train_step(inputs, model, criterion, optimizer, args):
    images, targets = inputs

    with tf.GradientTape() as tape:
        pred = model(images, mu_g=targets[..., :2], training=True)
        loss = criterion(targets, pred)
        loss = tf.math.reduce_mean(loss)
        loss += sum(model.losses)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(
        zip(gradients, model.trainable_variables)
    )
    acc = calc_coord_accuracy(
        targets, pred, args.DATASET.COMMON.OUTPUT_SHAPE
    )
    return loss, acc


@tf.function
def val_step(inputs, model, criterion, args):
    images, targets = inputs
    pred = model(images, training=False)
    loss = criterion(targets, pred)
    loss = tf.math.reduce_mean(loss)
    acc = calc_coord_accuracy(
        targets, pred, args.DATASET.COMMON.OUTPUT_SHAPE
    )
    return loss, acc
