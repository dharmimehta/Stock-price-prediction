#importing libraries
from sklearn.preprocessing import MinMaxScaler
#from sklearn.exceptions import DataConversionWarning
#warnings.filterwarnings(action='ignore', category=DataConversionWarning)
from keras.models import Sequential
from keras.layers import Dense, LSTM
#import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#reading the csv file
headers=['Date','Open','High','Low','Last','Close','TotalTradeQuantity','Turnover(Lacs)']
df = pd.read_csv('NSE-TATAGLOBAL11.csv')
print(df)
#plotting the historical data graph
df['Date'] = pd.to_datetime(df.Date)
df=df.sort_values(by='Date')
#df.ix[pd.to_datetime(df.Date).order().index]
print(df)
df = pd.read_csv('NSE-TATAGLOBAL11.csv')
df['Date'] = pd.to_datetime(df.Date)
df=df.sort_values(by='Date')
x = df['Date']
y = df['Close']

plt.plot(x,y)
plt.xlabel('date',fontsize=18)
plt.ylabel('close',fontsize=18)
plt.show()

#creating dataframe
data = df.sort_index(ascending=True, axis=0)
data['Date'] = pd.to_datetime(data.Date)
data=data.sort_values(by='Date')

new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])
for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]

new_data['Date'] = pd.to_datetime(new_data.Date)
new_data=new_data.sort_values(by='Date')


#setting index
new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)

#creating train and test sets
dataset = new_data.values

plt.plot(new_data['Close'])
train = dataset[0:987,:]
valid = dataset[987:,:]


#converting dataset into x_train and y_train and for normalizing data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)


x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# create and fit the LSTM network designing lstm model
model = Sequential()  #to ensure data is received sequentially to each layer
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))  #2D layer supports specification of input shape

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)




#predicting 246 values, using past 60 from the train data
inputs = new_data[len(new_data) - len(valid) - 60:].values

inputs = inputs.reshape(-1,1)


inputs  = scaler.transform(inputs)

X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)


X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)

closing_price = scaler.inverse_transform(closing_price)



#my data from here ---------------------
'''

'''
closing_price=closing_price.reshape(-1,1)

#for plotting

train = new_data[:987]
valid = new_data[987:]

#slen=len(valid['Close'])
valid['closing_price']=closing_price
#valid=valid.insert(slen,'closing_price',closing_price)

#valid = valid.assign(closing_price=pd.Series(np.random.randn(slen)).values)
#valid['closing_price'] = pd.Series(np.random.randn(slen), index=valid.index)

plt.plot(valid['Close'])
plt.plot(train['Close'])
plt.plot(valid[['Close','closing_price']])
plt.xlabel('date',fontsize=18)
plt.ylabel('close',fontsize=18)
