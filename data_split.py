import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

height = width = 48


def data_split():
    df = pd.read_csv('train.csv')
    unusual = []
    for i in range(df.shape[0]):
        image = np.fromstring(df.iloc[i, 1], sep=' ')
        image = image.astype(np.uint8)
        if np.max(np.bincount(image)) >= height * width / 3:
            # plt.imshow(image.reshape(height,width))
            # plt.show()
            unusual.append(i)
    print(len(unusual))
    df = df.drop(unusual)
    df.index = [i for i in range(df.shape[0])]
    train_data = df.sample(frac=0.9, random_state=0)
    val_data = df[~df.index.isin(train_data.index)]
    train_data.to_csv('saved/data/fer2013/train.csv', index=False)
    val_data.to_csv('saved/data/fer2013/val.csv', index=False)
