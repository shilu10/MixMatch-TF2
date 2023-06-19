def mixmatch(model, X, y, U, T, K, beta):
    batch_size = X.shape[0]
    # mean logits from augmentation of unlabelled data.
    mean_logits = guess_labels(U, model, K)
    # using the label smoothing technique for sharpening the probability dis.
    qb = sharpen(mean_logits, T)
    # repeat the probability dis multiple times(K)
    qb = tf.concat([qb for _ in range(K)], axis=0)
    # concatenate both labelled X and unlabelled X and lab_y and unlab_y
    U = tf.concat([_ for _ in U], axis=0)
    XU = tf.concat([X, U], axis=0)
    XUy = tf.concat([y, qb], axis=0)
    # shuffle the combined dataset.
    indices = tf.random.shuffle(tf.range(XU.shape[0]))
    W = tf.gather(XU, indices)
    Wy = tf.gather(XUy, indices)
    # and use the mixup data augmentation with the shuffled data and unshuffle data.
    XU, XUy = mixup(XU, W, XUy, Wy, beta=beta)
    XU = tf.split(XU, K + 1, axis=0)
    XU = interleave(XU, batch_size)
    return XU, XUy
    

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [tf.concat(v, axis=0) for v in xy]