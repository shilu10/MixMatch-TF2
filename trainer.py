def train(labelled_ds, unlabelled_ds, val_ds, epochs, **kwargs):
    # loss and metrics trackers
    xe_loss_tracker = tf.keras.metrics.Mean()
    l2_loss_tracker = tf.keras.metrics.Mean()
    total_loss_tracker = tf.keras.metrics.Mean()
    train_acc_tracker = tf.keras.metrics.Mean()
    val_loss_tracker = tf.keras.metrics.Mean()
    val_acc_tracker = tf.keras.metrics.Mean()
    
    # arguments
    K = kwargs.get("K")
    beta = kwargs.get('beta')
    T = kwargs.get("T")
    ema_decay_rate = kwargs.get('ema_decay_rate')
    weight_decay_rate = kwargs.get('weight_decay_rate')
    learning_rate = kwargs.get("learning_rate")
    lambda_u = kwargs.get("lambda_u")
    n_classes = kwargs.get("n_classes")
    ckpt_dir = kwargs.get('ckpt_dir')
    log_path = kwargs.get('log_path')
    
    # models and optimizer
    model = WideResNet(n_classes, depth=28, width=2)
    ema_model = WideResNet(n_classes, depth=28, width=2)
    ema_model.set_weights(model.get_weights())
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # checkpoints
    model_ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(model_ckpt, f'{ckpt_dir}/model', max_to_keep=3)
    # for ema model
    ema_ckpt = tf.train.Checkpoint(step=tf.Variable(0), net=ema_model)
    ema_manager = tf.train.CheckpointManager(ema_ckpt, f'{ckpt_dir}/ema', max_to_keep=3)
    
    # summary writers
    train_writer = tf.summary.create_file_writer(f'{log_path}/train')
    val_writer = tf.summary.create_file_writer(f'{log_path}/validation')
    
    model_ckpt.restore(manager.latest_checkpoint)
    ema_ckpt.restore(ema_manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
        
    for epoch in range(epochs):
        print(f'Epoch; {epoch}')
        for unlabelled_batch in tqdm(unlabelled_ds):
            model_ckpt.step.assign_add(1)
            ema_ckpt.step.assign_add(1)
            for i, labelled_batch in enumerate(labelled_ds):
                if i == 1:
                    break
                accuracy, xe_loss, l2_loss, total_loss = train_step(labelled_batch,
                                                            unlabelled_batch,
                                                            model,
                                                            ema_model,
                                                            optimizer,
                                                            xe_loss_tracker,
                                                            l2_loss_tracker, 
                                                            total_loss_tracker,
                                                            train_acc_tracker,
                                                            K=K,
                                                            T=T, 
                                                            beta=beta,
                                                            ema_decay_rate=ema_decay_rate,
                                                            weight_decay_rate=weight_decay_rate,
                                                            lambda_u=lambda_u
                                                            )
        
        for val_batch in val_ds:
            val_accuracy, val_loss = test_step(
                                        val_batch,
                                        model,
                                        val_acc_tracker,
                                        val_loss_tracker
                                    )
        with train_writer.as_default():
            tf.summary.scalar('xe_loss', xe_loss, step=epoch)
            tf.summary.scalar('l2u_loss', l2_loss, step=epoch)
            tf.summary.scalar('total_loss', total_loss, step=epoch)
            tf.summary.scalar('accuracy', accuracy, step=epoch)
        
        with val_writer.as_default():
            tf.summary.scalar('xe_loss', val_loss, step=epcoh)
            tf.summary.scalar('val_accuracy', val_accuracy, step=epoch)   
            
        if epoch % 10 == 0:
            model_save_path = manager.save(checkpoint_number=int(model_ckpt.step))
            ema_save_path = ema_manager.save(checkpoint_number=int(ema_ckpt.step))
            print(f'Saved model checkpoint for epoch {int(model_ckpt.step)} @ {model_save_path}')
            print(f'Saved ema checkpoint for epoch {int(ema_ckpt.step)} @ {ema_save_path}')
            
        print(f"train_loss: {total_loss}, xe_loss: {xe_loss}, l2_loss: {l2_loss}, train_accuracy: {accuracy}")
        print(f"val_loss: {val_loss}, val_accuracy: {val_accuracy}")
    
    for writer in [train_writer, val_writer]:
        writer.flush()