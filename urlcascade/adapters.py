"""Actual model adapters; imports are lazy so policy tests need no ML installs."""


def preprocess_link(link):
    normalized = link.lower()
    for marker in ('http://', 'https://', 'www.', ' '):
        normalized = normalized.replace(marker, '')
    return ''.join(c for c in normalized if c.isascii() and c.isalnum())


def vectorizer(vocabulary=None):
    import tensorflow as tf
    return tf.keras.layers.TextVectorization(
        standardize='lower_and_strip_punctuation', split='character', ngrams=(3,),
        output_mode='int', max_tokens=10000, output_sequence_length=50,
        vocabulary=vocabulary)


class SklearnT1:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        classes = list(pipeline.classes_)
        if set(classes) != {0, 1}:
            raise ValueError('T1_classes_must_be_y_mal_0_1')
        self.malicious_column = classes.index(1)

    def __call__(self, raw_features):
        return self.pipeline.predict_proba(raw_features)[:, self.malicious_column].tolist()


class KerasT2:
    def __init__(self, model, vocabulary):
        self.model = model
        self.vectorizer = vectorizer(vocabulary)

    def __call__(self, raw_urls, raw_features):
        import numpy as np
        import tensorflow as tf
        if not raw_urls:
            return []
        text = tf.constant([preprocess_link(u) for u in raw_urls])
        encoded = self.vectorizer(text)
        # Deliberately no StandardScaler here. .numpy() synchronizes result.
        return self.model([encoded, np.asarray(raw_features, dtype=np.float32)],
                          training=False).numpy().reshape(-1).tolist()
