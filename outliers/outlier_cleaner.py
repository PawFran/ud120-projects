#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    import numpy as np
    import pandas as pd
    
    cleaned_data = []

    ### your code goes here
    array = np.concatenate((ages, net_worths, predictions), axis = 1) 
    
    df = pd.DataFrame(array, columns = ['ages', 'net_worths', 'predictions'])
    df['prediction_error'] = (df['net_worths'] - df['predictions']).abs()
    df_sorted = df.sort_values('prediction_error')
    df_truncated = df_sorted.iloc[0:81] # take first 81 rows out of 90

    for row in df_truncated.iterrows():
	data = row[1]
    	cleaned_data.append((data[0], data[1], data[2]))

    return cleaned_data

