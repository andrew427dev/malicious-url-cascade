"""Experiment A. Train the pinned model, retaining only A's internal evaluation."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'upstream')]
from urlcascade.runtime import compute_context
from verify_assets import verify


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--runtime', choices=['kaggle','colab'], required=True)
    args = p.parse_args()
    compute_context(args.runtime, args.out)
    from sklearn.model_selection import train_test_split
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from Utils import DataProcessing, Model
    if tf.__version__ != '2.12.0':
        raise ValueError('expected TensorFlow 2.12.0')
    tf.keras.utils.set_random_seed(42)
    for device in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(device, True)
    args.out.mkdir(parents=True, exist_ok=False)
    config = json.loads((ROOT/'configs/upstream.json').read_text())
    frames = []
    for name in ('dataset/df_final.csv','dataset/all_top_domains_merged.csv'):
        path = args.assets/name
        if verify(path,config['files'][name])['status'] != 'verified':
            raise ValueError('source mismatch: ' + name)
        frame = pd.read_csv(path)
        if list(frame.columns) != ['url','type']:
            raise ValueError('expected two original columns')
        frame.index = [f'{path.name}:{i}' for i in range(2,len(frame)+2)]
        frames.append(frame)
    frame = pd.concat(frames).sample(frac=1,random_state=42).drop_duplicates().dropna()
    train, val = train_test_split(frame,test_size=0.2,stratify=frame['type'],random_state=42)
    ids = dict(train=train.index.tolist(),validation=val.index.tolist())
    (args.out/'source-ids.json').write_text(json.dumps(ids))
    for label, subset in [('train',train),('validation',val)]:
        print('A_SOURCE_ROWS',label,len(subset),dict(Counter(map(int,subset['type']))),flush=True)
    batch = 128
    train_ds = DataProcessing.process_train_data(train.copy(), batch)
    layer = tf.keras.layers.TextVectorization(standardize='lower_and_strip_punctuation',
        split='character',ngrams=(3,),max_tokens=10000,output_sequence_length=50,output_mode='int')
    layer.adapt(train_ds.map(lambda inputs,y: inputs[0]))
    (args.out/'vocab.json').write_text(json.dumps(layer.get_vocabulary()))
    train_ds = train_ds.map(lambda x,y: DataProcessing.vectorize_text(x[0],x[1],y,layer))
    train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
    val_ds = DataProcessing.process_data(val.copy(),batch,layer)
    model = Model.create_model(50,10000,18)
    assert model.count_params()==162481
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    settings=dict(experiment='A',source_commit=config['commit'],seed=42,batch_size=batch,epochs=20,
        source_before_original_preprocess={k:len(v) for k,v in ids.items()},
        source_ids_sha256=hashlib.sha256((args.out/'source-ids.json').read_bytes()).hexdigest(),
        started_utc=datetime.now(timezone.utc).isoformat(), python=sys.version, tensorflow=tf.__version__,
        gpu_devices=[d.name for d in tf.config.list_physical_devices('GPU')])
    (args.out/'settings.json').write_text(json.dumps(settings,indent=2))
    class Progress(tf.keras.callbacks.Callback):
        def on_epoch_begin(self,epoch,logs=None):
            self.started = time.monotonic()
        def on_epoch_end(self,epoch,logs=None):
            with (args.out/'progress.jsonl').open('a') as stream:
                row=dict(epoch=epoch+1,seconds=time.monotonic()-self.started,**(logs or {}))
                stream.write(json.dumps(row)+'\n')
                print('A_EPOCH',row,flush=True)
            if epoch == 0 and row['seconds'] * 20 * 1.5 > 4 * 3600:
                self.model.stop_training = True
                (args.out/'resource-review.json').write_text(json.dumps({
                    'first_epoch_seconds':row['seconds'],'limit_hours':4,
                    'projected_hours_with_margin':row['seconds']*20*1.5/3600}))
    model.fit(train_ds,validation_data=val_ds,epochs=20,verbose=2,callbacks=[Progress(),
        tf.keras.callbacks.CSVLogger(str(args.out/'epochs.csv')),
        tf.keras.callbacks.ModelCheckpoint(str(args.out/'latest.h5'))])
    if (args.out/'resource-review.json').exists():
        raise RuntimeError('full first epoch exceeded the planned four-hour budget')
    # The final epoch is the reproduction result. Read labels alongside scores in the same pass.
    cm = np.zeros((2,2),dtype=int)
    for inputs,y_safe in val_ds:
        y = 1-y_safe.numpy().astype(int).reshape(-1)
        pred = (model(inputs,training=False).numpy().reshape(-1)<0.5).astype(int)
        np.add.at(cm,(y,pred),1)
    report=dict(experiment='A',epoch=20,internal_confusion_normal_malicious=cm.tolist(),
                validation_rows=int(cm.sum()),accuracy=float(np.trace(cm)/cm.sum()),
                malicious_recall=float(cm[1,1]/cm[1].sum()),
                external_evaluation='deferred until B models and gate frozen',
                model_sha256=hashlib.sha256((args.out/'latest.h5').read_bytes()).hexdigest())
    (args.out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
