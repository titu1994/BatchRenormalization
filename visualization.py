import seaborn as sns
sns.set_style('whitegrid')

import json


with open('history/batchnorm_history.txt') as f, open('history/batch_renorm_mode_0_history.txt') as f2, \
    open('history/batch_renorm_mode_2_history.txt') as f3:
    history_bn = json.load(f)
    history_renorm_mode_0 = json.load(f2)
    history_renorm_mode_2 = json.load(f3)

sns.plt.subplot(2, 2, 1)
sns.plt.plot(history_bn['loss'], label='batchnorm training loss')
sns.plt.plot(history_renorm_mode_0['loss'], label='renorm training loss')
sns.plt.plot(history_bn['val_loss'], label='batchnorm validation loss')
sns.plt.plot(history_renorm_mode_0['val_loss'], label='renorm validation loss')
sns.plt.title('Renorm Mode 0 comparison')
sns.plt.legend()

sns.plt.subplot(2, 2, 2)
sns.plt.plot(history_bn['acc'], label='batchnorm training accuracy')
sns.plt.plot(history_renorm_mode_0['acc'], label='renorm training accuracy')
sns.plt.plot(history_bn['val_acc'], label='batchnorm validation accuracy')
sns.plt.plot(history_renorm_mode_0['val_acc'], label='renorm validation accuracy')
sns.plt.title('Renorm Mode 0 comparison')
sns.plt.legend(loc='lower right')

sns.plt.subplot(2, 2, 3)
sns.plt.plot(history_bn['loss'], label='batchnorm training loss')
sns.plt.plot(history_renorm_mode_2['loss'], label='renorm training loss')
sns.plt.plot(history_bn['val_loss'], label='batchnorm validation loss')
sns.plt.plot(history_renorm_mode_2['val_loss'], label='renorm validation loss')
sns.plt.title('Renorm Mode 2 comparison')
sns.plt.legend()

sns.plt.subplot(2, 2, 4)
sns.plt.plot(history_bn['acc'], label='batchnorm training accuracy')
sns.plt.plot(history_renorm_mode_2['acc'], label='renorm training accuracy')
sns.plt.plot(history_bn['val_acc'], label='batchnorm validation accuracy')
sns.plt.plot(history_renorm_mode_2['val_acc'], label='renorm validation accuracy')
sns.plt.title('Renorm Mode 2 comparison')
sns.plt.legend(loc='lower right')


sns.plt.show()