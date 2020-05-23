import tensorflow as tf


def weighted_loss(original_loss_func, weights_list):

    @tf.function
    def loss_func(true, pred):

        axis = -1  # if channels last
        # axis=  1 #if channels first

        # argmax returns the index of the element with the greatest value
        # done in the class axis, it returns the class index
        classSelectors = tf.argmax(true, axis=axis, output_type=tf.int32)

        # considering weights are ordered by class, for each class
        # true(1) if the class index is equal to the weight index
        classSelectors = [tf.equal(i, classSelectors)
                          for i in range(len(weights_list))]

        # casting boolean to float for calculations
        # each tensor in the list contains 1 where ground true class is equal to its index
        # if you sum all these, you will get a tensor full of ones.
        classSelectors = [tf.cast(x, tf.float32) for x in classSelectors]

        # for each of the selections above, multiply their respective weight
        weights = [sel * w for sel, w in zip(classSelectors, weights_list)]

        # sums all the selections
        # result is a tensor with the respective weight for each element in predictions
        weight_multiplier = weights[0]
        for i in range(1, len(weights)):
            weight_multiplier = weight_multiplier + weights[i]

        # make sure your original_loss_func only collapses the class axis
        # you need the other axes intact to multiply the weights tensor
        loss = original_loss_func(true, pred)
        loss = loss * weight_multiplier

        return loss
    return loss_func
