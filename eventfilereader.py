import tensorflow as tf

logdir = 'runs/detect/train14/events.out.tfevents.1733957220.DESKTOP-G3K96A3.16500.0'
for event in tf.compat.v1.train.summary_iterator(logdir):
    for value in event.summary.value:
        print(f"Tag: {value.tag}, Value: {value.simple_value}")
