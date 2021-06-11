import pandas as pd

"""Handling categorical values"""


def change_to_categories(dataframe):
    """If its a string change it into category. Doesn't work for date and time, use process_dates() for dates"""
    for label, content in dataframe.items():
        if pd.api.types.is_string_dtype(content):
            dataframe[label] = dataframe[label].astype('category').cat.as_ordered()


def change_to_categories_test(dataframe1, dataframe2):
    """Change the dataframe2 content as categories(dataframe1 categories)"""
    for label, content in dataframe2.items():
        try:
            if dataframe1[label].dtype.name == 'category':
                dataframe2[label] = pd.Categorical(content, categories=dataframe1[label].cat.categories, ordered=True)
        except:
            print("Error while converting check the dataframes labels")


def numericalise_categories(df, max_n_cat):
    for col_name, _ in df.items():
        #print(col_name)
        if not pd.api.types.is_numeric_dtype(df[col_name]) and not pd.api.types.is_string_dtype(df[col_name]):
            if (max_n_cat is None or len(df[col_name].cat.categories) > max_n_cat) :
                df[col_name] = pd.Categorical(df[col_name]).codes + 1
            elif not pd.api.types.is_numeric_dtype(df[col_name]) and (len(df[col_name].cat.categories) < max_n_cat):
                print(col_name)
                df = pd.concat([df, pd.get_dummies(df[col_name])], axis =1)
                df.drop([col_name], axis =1, inplace= True)
    
    return df

"""Handling null values"""


def remove_rows_with_null(dataframe, label):
    dataframe.drop(dataframe.loc[dataframe[label].isnull(), :].index, axis=0, inplace=True)


def handle_null_values_train(dataframe):
    columns_null = {}
    for label, contents in dataframe.items():
        if dataframe[label].isnull().any() and pd.api.types.is_numeric_dtype(dataframe[label]):
            median = contents.median()
            dataframe[label] = dataframe[label].fillna(median)
            columns_null[label] = contents.median()
    return columns_null


def handle_null_values_test(dataframe, columns_null):
    for label, contents in dataframe.items():
        if dataframe[label].isnull().any() and pd.api.types.is_numeric_dtype(dataframe[label]):
            dataframe[label] = dataframe[label].fillna(columns_null[label])


"""Handling numericals"""


def scale_numericals_train(dataframe, scaler_name='RobustScaler', cols=None):
    scaler_module = __import__('sklearn.preprocessing', fromlist=scaler_name)

    columns_scaler = {}
    if cols == None:
        for labels, content in dataframe.items():
            scaler = getattr(scaler_module, str(scaler_name))()
            if pd.api.types.is_float_dtype(dataframe[labels]) or pd.api.types.is_int64_dtype(dataframe[labels]):
                dataframe[labels] = content.fillna(content.median())
                dataframe[labels] = scaler.fit_transform(content.to_numpy().reshape(-1, 1))
                columns_scaler[labels] = scaler
    else:
        for labels, content in dataframe[cols].items():
            scaler = getattr(scaler_module, str(scaler_name))()
            if pd.api.types.is_float_dtype(dataframe[labels]) or pd.api.types.is_int64_dtype(dataframe[labels]):
                dataframe[labels] = content.fillna(content.median())
                dataframe[labels] = scaler.fit_transform(content.to_numpy().reshape(-1, 1))
                columns_scaler[labels] = scaler

        return columns_scaler


def scale_numericals_test(dataframe1, columns_scaler, cols=None):
    try:
        if cols is not None:
            dataframe = dataframe1[cols]
            
        for labels, content in dataframe.items():
            if pd.api.types.is_float_dtype(dataframe[labels]) or pd.api.types.is_int64_dtype(dataframe[labels]) and cols == None:
                dataframe[labels] = columns_scaler[labels].transform(content.to_numpy().reshape(-1, 1))
            elif cols:
                for labels, content in dataframe.items():
                    if pd.api.types.is_float_dtype(dataframe[labels]) or pd.api.types.is_int64_dtype(dataframe[labels]):
                        dataframe[labels] = columns_scaler[labels].transform(content.to_numpy().reshape(-1, 1))
        
        dataframe1[cols] = dataframe[cols]
    except:
        print(f'{labels} error here')



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

    dataframe[fieldName + '_elapsed_days'] = (data - data.min()).dt.days
    dataframe.drop(fieldName, axis=1, inplace=True)


"""Getting null percentages"""


def get_null_percentage(dataframe):
    print((dataframe.isnull().sum() / len(dataframe)) * 100)
