"""Train one new upstream GRU per experiment and seal the selected checkpoint."""

import argparse

from inputs import ROOT, sha, write_json, np, tf
from Utils.Model import create_model


def dataset(out, part, shuffle):
    tokens = np.load(out / (part + "-tokens.npy"))
    x = np.load(out / (part + "-features.npy"))
    import pandas as pd

    labels = pd.read_csv(out / (part + "-rows.csv.gz")).type.to_numpy()
    ds = tf.data.Dataset.from_tensor_slices(((tokens, x), labels))
    if shuffle == "epoch":
        ds = ds.shuffle(len(labels), seed=42, reshuffle_each_iteration=True)
    elif shuffle == "evaluation":
        ds = ds.shuffle(10000, seed=42)
    ds = ds.batch(128)
    return ds.cache().prefetch(1) if shuffle == "evaluation" else ds.prefetch(1)


def train(kind):
    out = ROOT / kind
    if any(
        (out / name).exists()
        for name in ("trained.json", "epochs.csv", "gru.h5", "training-inputs.json")
    ):
        raise FileExistsError("Prior training attempt exists")
    from provenance import seal

    seal(kind)
    tf.keras.utils.set_random_seed(42)
    author = kind.startswith("author-")
    training = dataset(out, "train", "none" if author else "epoch")
    valid = None if kind == "author-final" else dataset(out, "validation", "evaluation")
    model = create_model(50, 10000, 18)
    assert model.count_params() == 162481
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    callbacks = [tf.keras.callbacks.CSVLogger(str(out / "epochs.csv"))]
    # h5py supports the Korean workspace path even with old TensorFlow.
    best = float("inf")

    class Save(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            nonlocal best
            if (author and epoch == 19) or (not author and logs["val_loss"] < best):
                self.model.save_weights(str(out / "gru.h5"))
                best = logs.get("val_loss", logs["loss"])

    history = model.fit(
        training,
        validation_data=valid,
        epochs=20,
        verbose=2,
        callbacks=[*callbacks, Save()],
    )
    if (
        len(history.history["loss"]) != 20
        or not np.isfinite(history.history["loss"]).all()
    ):
        raise RuntimeError("Incomplete training")
    write_json(
        out / "trained.json",
        {
            "history": history.history,
            "seed": 42,
            "epoch": 20 if author else int(np.argmin(history.history["val_loss"])) + 1,
            "params": model.count_params(),
            "weights_sha256": sha(out / "gru.h5"),
            "vocabulary_sha256": sha(out / "vocab.json"),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("kind")
    train(parser.parse_args().kind)
