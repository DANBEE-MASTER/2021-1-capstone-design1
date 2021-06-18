from tensorflow.python.client import device_lib
device_lib.list_local_devices()

import os
import numpy as np
import pandas as pd
from glob import glob
from itertools import chain
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score, average_precision_score
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf

DATA_DIR = ''
image_size = 128
batch_size = 128


train_df = pd.read_csv(f'{DATA_DIR}Data_Entry_2017-train_df.csv')
valid_df = pd.read_csv(f'{DATA_DIR}Data_Entry_2017-valid_df.csv')
test_df = pd.read_csv(f'{DATA_DIR}Data_Entry_2017-test_df.csv')



data_image_paths = {os.path.basename(x): x for x in glob(os.path.join(DATA_DIR, 'images*', '*', '*.png'))}
data_image_paths = {os.path.basename(x): x for x in glob(os.path.join(DATA_DIR, 'images*', '*', '*.png'))}
data_image_paths = {os.path.basename(x): x for x in glob(os.path.join(DATA_DIR, 'images*', '*', '*.png'))}



train_df['path'] = train_df['Image Index'].map(data_image_paths.get)
valid_df['path'] = valid_df['Image Index'].map(data_image_paths.get)
test_df['path'] = test_df['Image Index'].map(data_image_paths.get)


labels = np.unique(list(chain(*train_df['Finding Labels'].map(lambda x: x.split('|')).tolist())))
labels = np.unique(list(chain(*valid_df['Finding Labels'].map(lambda x: x.split('|')).tolist())))
labels = np.unique(list(chain(*test_df['Finding Labels'].map(lambda x: x.split('|')).tolist())))
labels = [x for x in labels if len(x) > 0]


all_labels = labels


for label in labels:
    if len(label) > 1:
        train_df[label] = train_df['Finding Labels'].map(lambda finding: 1.0 if label in finding else 0.0)
        valid_df[label] = valid_df['Finding Labels'].map(lambda finding: 1.0 if label in finding else 0.0)
        test_df[label] = test_df['Finding Labels'].map(lambda finding: 1.0 if label in finding else 0.0)
        
        
        
        
from sklearn.utils import shuffle

train_df= shuffle(train_df, random_state=2021)
valid_df= shuffle(valid_df, random_state=2021)
test_df= shuffle(test_df, random_state=2021)


print(f'train_df {train_df.shape[0]} valid_df {valid_df.shape[0]} test_df {test_df.shape[0]}')


print(train_df['Finding Labels'].value_counts())
print(valid_df['Finding Labels'].value_counts())
print(test_df['Finding Labels'].value_counts())


train_df['labels'] = train_df.apply(lambda x: x['Finding Labels'].split('|'), axis=1)
valid_df['labels'] = valid_df.apply(lambda x: x['Finding Labels'].split('|'), axis=1)
test_df['labels'] = test_df.apply(lambda x: x['Finding Labels'].split('|'), axis=1)


core_idg = ImageDataGenerator(rescale=1 / 255)

train_gen = core_idg.flow_from_dataframe(dataframe=train_df,
                                             directory=None,
                                             x_col='path',
                                             y_col='labels',
                                             class_mode='categorical',
                                             batch_size=batch_size,
                                             classes=labels,
                                             target_size=(image_size, image_size))

valid_gen = core_idg.flow_from_dataframe(dataframe=valid_df,
                                             directory=None,
                                             x_col='path',
                                             y_col='labels',
                                             class_mode='categorical',
                                             batch_size=batch_size,
                                             classes=labels,
                                             target_size=(image_size, image_size))

test_gen = core_idg.flow_from_dataframe(dataframe=test_df,
                                             directory=None,
                                             x_col='path',
                                             y_col='labels',
                                             class_mode='categorical',
                                             batch_size=batch_size,
                                             classes=labels,
                                             target_size=(image_size, image_size))


valid_X, valid_Y = next(core_idg.flow_from_dataframe(dataframe=valid_df,
                                                       directory=None,
                                                       x_col='path',
                                                       y_col='labels',
                                                       class_mode='categorical',
                                                       batch_size=1024,
                                                       classes=labels,
                                                       target_size=(image_size, image_size)))

test_X, test_Y = next(core_idg.flow_from_dataframe(dataframe=test_df,
                                                       directory=None,
                                                       x_col='path',
                                                       y_col='labels',
                                                       class_mode='categorical',
                                                       batch_size=1024,
                                                       classes=labels,
                                                       target_size=(image_size, image_size)))



import tensorflow as tf
from tensorflow.keras.metrics import AUC
import tensorflow.keras


strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    
    model = tf.keras.applications.DenseNet121(input_shape=(image_size, image_size, 3), weights=None, include_top=True, classes=2)
    model.summary()
    
    
    model.compile(optimizer='adam', 
                  loss = 'binary_crossentropy', 
                  metrics = [AUC(name='AUC'), 'accuracy'])
    
    
    
    
    history = model.fit(train_gen,
                        epochs=50,
                        validation_data=(valid_X, valid_Y),)



history_df = pd.DataFrame(history.history)
history_df.to_csv('history_df.csv')




history_df.loc[:, ['loss', 'val_loss']].plot();
plt.savefig('loss,val_loss.png')
history_df.loc[:, ['accuracy', 'val_accuracy']].plot();
plt.savefig('accuracy,val_accuracy.png')
print("Minimum validation loss: {}".format(history_df['val_loss'].min()))
print("Maximum validation accuracy: {}".format(history_df['val_accuracy'].max()))


for c_label, s_count in zip(all_labels, 100*np.mean(valid_Y,0)):
    print('%s: %2.2f%%' % (c_label, s_count))
    
    
    
valid_pred_Y = model.predict(valid_X, batch_size = 32, verbose = True)



from sklearn.metrics import roc_curve, auc
fig, c_ax = plt.subplots(1,1, figsize = (9, 9))
for (idx, c_label) in enumerate(all_labels):
    fpr, tpr, thresholds = roc_curve(valid_Y[:,idx].astype(int), valid_pred_Y[:,idx])
    c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))
c_ax.legend()
c_ax.set_xlabel('False Positive Rate')
c_ax.set_ylabel('True Positive Rate')
fig.savefig('valid_trained_net.png')


from sklearn.metrics import roc_auc_score
# ROC AUC
auc = roc_auc_score(valid_Y, valid_pred_Y)
print('ROC AUC: %f' % auc)




for c_label, s_count in zip(all_labels, 100*np.mean(test_Y,0)):
    print('%s: %2.2f%%' % (c_label, s_count))
    
    
test_pred_Y = model.predict(test_X, batch_size = 32, verbose = True)


from sklearn.metrics import roc_curve, auc
fig, c_ax = plt.subplots(1,1, figsize = (9, 9))
for (idx, c_label) in enumerate(all_labels):
    fpr, tpr, thresholds = roc_curve(test_Y[:,idx].astype(int), test_pred_Y[:,idx])
    c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))
c_ax.legend()
c_ax.set_xlabel('False Positive Rate')
c_ax.set_ylabel('True Positive Rate')
fig.savefig('test_trained_net.png')


from sklearn.metrics import roc_auc_score
# ROC AUC
auc = roc_auc_score(test_Y, test_pred_Y)
print('ROC AUC: %f' % auc)




# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(test_X, test_Y, batch_size=128, return_dict=True)
print("test loss, test acc:", results)

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
print("Generate predictions for 3 samples")
predictions = model.predict(test_X[:3])
print("predictions shape:", predictions.shape)


predictions=pd.DataFrame(predictions)
predictions.to_csv("predictions.csv")



test_df.to_csv("test_df.csv")


model.save('model.h5')
model.save_weights('model_weights.h5')