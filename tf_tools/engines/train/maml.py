import os
import numpy as np
import tensorflow as tf


def train(model=None, epochs=10, batch_size=32, format_paths=True, x_train=None, y_train=None,
          train_gen=None, train_size=None, val_gen=None, val_size=None, train_datagen=None,
          train_loss=None, train_accuracy=None, test_loss=None, test_accuracy=None,
          val_loss=None, val_accuracy=None, train_step=None, test_step=None, meta_step=None,
          checkpoint_path=None, max_patience=25, nb_classes=None,
          train_summary_writer=None, val_summary_writer=None, csv_output_file=None,
          optimizer=None, meta_optimizer=None, loss_object=None, lr=0.001,
          train_epochs=3, n_tasks=1, **kwargs):

    min_loss = 100
    min_loss_acc = 0
    patience = 0

    # shuffle the meta train images maintaining the same label order
    index_sets = [np.argwhere(i == y_train) for i in np.unique(y_train)]
    x_meta_train = np.copy(x_train)
    for class_indexes in index_sets:
        shuffled_class_indexes = np.copy(class_indexes)
        np.random.shuffle(shuffled_class_indexes)
        for i in range(len(class_indexes)):
            x_meta_train[class_indexes[i]] = x_train[shuffled_class_indexes[i]]

    meta_train_gen = train_datagen.flow(x_meta_train, y_train, batch_size=batch_size, seed=42)

    results = 'epoch,loss,accuracy,val_loss,val_accuracy\n'

    if not meta_optimizer:
        meta_optimizer = tf.keras.optimizers.Adam()

    for epoch in range(epochs):

        for train_epoch in range(train_epochs):

            batches = 0
            while batches < train_size / batch_size:

                batches += n_tasks

                # get the weights of the initial model that will do the meta learning
                meta_model_weights = model.get_weights()

                for k in range(n_tasks):

                    # train on the task (one batch)
                    images, labels = train_gen.next()
                    train_step(images, labels)

                    # test on the validation set the improvement achieved on one task for the meta learning
                    sum_gradients = np.zeros_like(model.trainable_variables)
                    images, labels = meta_train_gen.next()
                    gradients = meta_step(images, labels)
                    gradients = np.array([np.array(x) for x in gradients])
                    sum_gradients = sum_gradients + gradients

                    # set weights of the model to the weights of the original model
                    model.set_weights(meta_model_weights)

            # update the weights of the meta learning model using the loss obtained from testing
            meta_optimizer.apply_gradients(
                zip(gradients, model.trainable_variables))

        # get the weights of the initial model that will do the meta learning
        meta_model_weights = model.get_weights()

        # train on the task (one epoch)
        batches = 0
        for images, labels in train_gen:
            train_step(images, labels)
            batches += 1
            if batches >= train_size / batch_size:
                # we need to break the loop by hand because
                # the generator loops indefinitely
                break

        # test the newly trained model on the training set
        batches = 0
        for val_images, val_labels in val_gen:
            test_step(val_images, val_labels)
            batches += 1
            if batches >= val_size / batch_size:
                # we need to break the loop by hand because
                # the generator loops indefinitely
                break

        # set weights of the model to the weights of the original model
        model.set_weights(meta_model_weights)

        results += '{},{},{},{},{}\n'.format(
            epoch,
            train_loss.result(),
            train_accuracy.result()*100,
            val_loss.result(),
            val_accuracy.result()*100)
        print('Epoch: {}, Train Loss: {}, Train Acc:{}, Test Loss: {}, Test Acc: {}'.format(
            epoch,
            train_loss.result(),
            train_accuracy.result()*100,
            val_loss.result(),
            val_accuracy.result()*100))

        if (val_loss.result() < min_loss):
            min_loss = val_loss.result()
            min_loss_acc = val_accuracy.result()
            patience = 0
            # serialize weights to HDF5
            if format_paths:
                checkpoint_path = checkpoint_path.format(
                    epoch=epoch, val_loss=min_loss, val_accuracy=min_loss_acc)
            model.save_weights(checkpoint_path)
        else:
            patience += 1

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)
            train_loss.reset_states()
            train_accuracy.reset_states()

        with val_summary_writer.as_default():
            tf.summary.scalar('loss', val_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', val_accuracy.result(), step=epoch)
            val_loss.reset_states()
            val_accuracy.reset_states()

        if patience >= max_patience:
            break

    file = open(csv_output_file, 'w')
    file.write(results)
    file.close()

    return min_loss, min_loss_acc
