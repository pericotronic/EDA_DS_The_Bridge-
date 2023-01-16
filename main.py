import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import pandas_ta as ta
data = yf.download(tickers = '^RUI', start = '2012-03-11',end = '2023-01-11')
data.head(10)

# Añadiendo indicadores
data['RSI']=ta.rsi(data.Close, length=15)
data['EMAF']=ta.ema(data.Close, length=20)
data['EMAM']=ta.ema(data.Close, length=100)
data['EMAS']=ta.ema(data.Close, length=150)

data['Target'] = data['Adj Close']-data.Open
data['Target'] = data['Target'].shift(-1)

data['TargetClass'] = [1 if data.Target[i]>0 else 0 for i in range(len(data))]

data['TargetNextClose'] = data['Adj Close'].shift(-1)

data.dropna(inplace=True)
data.reset_index(inplace = True)
data.drop(['Volume', 'Close', 'Date'], axis=1, inplace=True)

data_set = data.iloc[:, 0:11]#.values
pd.set_option('display.max_columns', None)

data_set.head(20)
#print(data_set.shape)
#print(data.shape)
#print(type(data_set))

results_to_predict=data_set.loc[:, ['TargetNextClose']]
results_to_predict


from sklearn.preprocessing import MinMaxScaler
prediction_scaler = MinMaxScaler(feature_range=(0,1))
prediction_scaler.fit(results_to_predict)
print(prediction_scaler)



from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
data_set_scaled = sc.fit_transform(data_set)
print(data_set_scaled)




# característica múltiple de los datos proporcionados al modelo
X = []
#print(data_set_scaled[0].size)
#data_set_scaled=data_set.values
backcandles = 8
print(data_set_scaled.shape[0])
for j in range(8):#data_set_scaled[0].size):#2 columns are target not X
    X.append([])
    for i in range(backcandles, data_set_scaled.shape[0]):#backcandles+2
        X[j].append(data_set_scaled[i-backcandles:i, j])

'''Aqui muevo el eje 0 (sublistas) a la posición 2 (tercera dimensión) en la variable X.
 Esto se hace para ajustar la forma del conjunto de datos de entrada X al requerimiento de una entrada de dos dimensiones del modelo LSTM.'''
X=np.moveaxis(X, [0], [2])

#borrar los primeros elementos de Y, debido a los backcandles para que coincidan con la longitud de X
#del(yi[0:backcandles])
#X, yi = np.array(X), np.array(yi)
# Elegir -1 para la última columna, clasificación si no -2...
X, yi =np.array(X), np.array(data_set_scaled[backcandles:,-3])
y=np.reshape(yi,(len(yi),1))
#y=sc.fit_transform(yi)
#X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
print(X)
print(X.shape)
print(y)
print(y.shape)




# split data into train test sets
splitlimit = int(len(X)*0.8)
print(splitlimit)
X_train, X_test = X[:splitlimit], X[splitlimit:]
y_train, y_test = y[:splitlimit], y[splitlimit:]
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print(y_train)




from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))






from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import TimeDistributed

import tensorflow as tf
import keras
from keras import optimizers
from keras.callbacks import History
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
import numpy as np
#tf.random.set_seed(20)
np.random.seed(10)

lstm_input = Input(shape=(backcandles, 8), name='lstm_input')
inputs =LSTM(8, name='first_layer')(lstm_input)
inputs = Dense(1, name='dense_layer')(inputs)
output = Activation('linear', name='output')(inputs)
model = Model(inputs=lstm_input, outputs=output)
adam = optimizers.Adam()
model.compile(optimizer=adam, loss='mse')
model.fit(x=X_train, y=y_train, batch_size=15, epochs=30, shuffle=True, validation_split = 0.1)




y_pred = model.predict(X_test)
#y_pred=np.where(y_pred > 0.43, 1,0)
for i in range(10):
    print(y_pred[i], y_test[i])



def price_predictedion():
    price_predicted=prediction_scaler.inverse_transform(y_pred)
    #print(price_predicted)
    return price_predicted

price_predicted=prediction_scaler.inverse_transform(y_pred)
dataframe_of_unscaled_data=pd.DataFrame(price_predicted)
print(dataframe_of_unscaled_data)
    


def actual_price():
    price_predicted=prediction_scaler.inverse_transform(y_test)
    #print(price_predicted)
    return price_predicted

price_actual=prediction_scaler.inverse_transform(y_test)
dataframe1=pd.DataFrame(price_actual)
dataframe1.tail()








plt.figure(figsize=(16,8))
predicted_prices = price_predictedion()
actual_prices = actual_price()
plt.plot(predicted_prices, label='Predicted Price')
plt.plot(actual_prices, label='Actual Price')
#plt.xlabel('Longitud_jornada')
#plt.ylabel('Price')
plt.legend()
plt.title('datos_des_escalados')
plt.show()



plt.figure(figsize=(16,8))
plt.plot(y_test, color = 'black', label = 'Test')
plt.plot(y_pred, color = 'green', label = 'pred')
plt.xlabel('Longitud_jornada')
plt.ylabel('Price')
plt.legend()
plt.title('datos_sin_desescalar')
plt.show()








    