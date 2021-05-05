import pandas as pd

"""Handling categorical values"""
def change_to_categories(dataframe):
    """If its a string change it into category. Doesn't work for date and time, use process_dates() for dates"""
    for label, content in dataframe.items():
        if pd.api.types.is_string_dtype(content):
            #print(content)
            dataframe[label] = dataframe[label].astype('category').cat.as_ordered()

def handle_categories(dataframe1, dataframe2):
    """Change the dataframe2 content as categories(dataframe1 categories)"""
    for label, content in dataframe2.items():
        try:
            if dataframe1[label].dtype.name == 'category':
                dataframe2[label] = pd.Categorical(content, categories=dataframe1[label].cat.categories, ordered=True)
        except:
            print("Error while converting check the dataframes labels")

def numericalise_categories(df, col_name, max_n_cat):
    if not pd.api.types.is_numeric_dtype(df[col_name]) and ( max_n_cat is None or len(df[col_name].cat.categories)>max_n_cat):
        df[col_name] = pd.Categorical(df[col_name]).codes+1


"""Handling null values"""
def remove_rows_with_null(dataframe, label):
    dataframe.drop(dataframe.loc[dataframe[label].isnull(), :].index, axis=0, inplace=True)


def handle_null_values(dataframe):
    for label, contents in dataframe.items():
        if dataframe[label].isnull().any():
            dataframe[label] = dataframe[label].fillna(contents.median())

"""Handling numericals"""
def handle_numericals(dataframe, scaler_name='RobustScaler'):
    scaler_module = __import__('sklearn.preprocessing', fromlist= scaler_name)
    scaler = getattr(scaler_module, scaler_name)
    for labels, content in dataframe.items():
        if pd.api.types.is_float_dtype(dataframe[labels]) or pd.api.types.is_int64_dtype(dataframe[labels]):
            dataframe[labels] = content.fillna(content.median())
            dataframe[labels] = scaler().fit_transform(content.to_numpy().reshape(-1,1))

"""Handling dates"""
def handle_dates(dataframe, fieldName):
    dataframe[fieldName] = pd.to_datetime(dataframe[fieldName])
    data = dataframe[fieldName]
    attributesList = ['year', 'month', 'day', 'dayofyear', 'dayofweek', \
                      'quarter', 'is_month_start', 'is_month_end', \
                      'is_quarter_start', 'is_quarter_end', 'is_year_start', 'is_year_end']
    for label in attributesList:
        dataframe[fieldName + '_' + label] = getattr(data.dt, label)
        if label == 'dayofweek':
            weekday = lambda x: x < 5
            dataframe[fieldName + '_weekday'] = weekday(getattr(data.dt, label))

    dataframe[fieldName + '_elapsed_days'] = (data - data.min())
    dataframe.drop(fieldName, axis=1, inplace=True)