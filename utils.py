def ema_weight_update(model, ema_model, ema_decay):
    ema_vars = ema_model.get_weights()
    model_vars = model.get_weights()
    
    if model_vars:
        for i in range(len(ema_vars)):
            ema_vars[i] = (1 - ema_decay) * model_vars[i] + ema_decay * ema_var[i]
    
    ema_model.set_weight(ema_vars)


def weight_decay(model, weight_decay):
    model_vars = model.get_weights()
    for i in range(len(model_vars)):
        model_vars[i] = model_vars[i] * (1 - weight_decay)
    
    model.set_weight(model_vars)


def semi_loss(labels_x, logits_x, labels_u, logits_u):
    xe_loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    loss_xe = xe_loss_func(labels, logits_x)
   
    loss_l2u = tf.square(labels_u - tf.nn.softmax(logits_u))
    loss_l2u = tf.reduce_mean(loss_l2u)
    return loss_xe, loss_l2u