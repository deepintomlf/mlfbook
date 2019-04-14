from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer

def data_scaler(data,scaler_type):
    if scaler_type == 'standard':
        transformed_data = StandardScaler().fit_transform(data)
    elif scaler_type == 'minmax':
        transformed_data = MinMaxScaler().fit_transform(data)
    elif scaler_type == 'normalize':
        transformed_data = Normalizer().fit_transform(data)
    else:
        transformed_data = None
    return transformed_data

