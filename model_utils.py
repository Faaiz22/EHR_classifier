
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence

def train_model(data, test_name):
    try:
        X = data[['AGE', 'SEX_M']].copy()
        y = data[test_name].copy()
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        return model
    except Exception as e:
        raise Exception(f"Error training model for {test_name}: {str(e)}")

def get_prediction(model, age, sex_numeric):
    try:
        X_new = np.array([[1, age, sex_numeric]])
        prediction = model.predict(X_new)[0]
        return prediction
    except Exception as e:
        raise Exception(f"Error making prediction: {str(e)}")

def calculate_clinical_reference_range(data, test_name, age, sex):
    try:
        age_groups = [
            (0, 18, "Pediatric"),
            (18, 30, "Young Adult"),
            (30, 50, "Adult"),
            (50, 65, "Middle Age"),
            (65, 120, "Elderly")
        ]
        age_group_info = None
        for min_age, max_age, group_name in age_groups:
            if min_age <= age < max_age:
                age_group_info = (min_age, max_age, group_name)
                break
        if age_group_info is None:
            raise ValueError(f"Age {age} is outside valid range")
        min_age, max_age, group_name = age_group_info
        age_sex_data = data[
            (data['SEX'] == sex) &
            (data['AGE'] >= min_age) &
            (data['AGE'] < max_age)
        ][test_name].dropna()
        if len(age_sex_data) < 10:
            age_sex_data = data[
                (data['SEX'] == sex) &
                (data['AGE'] >= max(0, min_age - 10)) &
                (data['AGE'] < min(120, max_age + 10))
            ][test_name].dropna()
        if len(age_sex_data) < 5:
            age_sex_data = data[data['SEX'] == sex][test_name].dropna()
        lower_bound = age_sex_data.quantile(0.025)
        upper_bound = age_sex_data.quantile(0.975)
        mean_val = age_sex_data.mean()
        median_val = age_sex_data.median()
        std_val = age_sex_data.std()
        reference_range = {
            'lower': lower_bound,
            'upper': upper_bound,
            'mean': mean_val,
            'median': median_val,
            'std': std_val,
            'age_group': group_name,
            'sample_size': len(age_sex_data),
            'percentile': None
        }
        return reference_range
    except Exception as e:
        raise Exception(f"Error calculating clinical reference range: {str(e)}")

def calculate_percentile(data, test_name, age, sex, test_value):
    try:
        reference_range = calculate_clinical_reference_range(data, test_name, age, sex)
        age_groups = [
            (0, 18, "Pediatric"),
            (18, 30, "Young Adult"),
            (30, 50, "Adult"),
            (50, 65, "Middle Age"),
            (65, 120, "Elderly")
        ]
        for min_age, max_age, group_name in age_groups:
            if min_age <= age < max_age:
                break
        age_sex_data = data[
            (data['SEX'] == sex) &
            (data['AGE'] >= min_age) &
            (data['AGE'] < max_age)
        ][test_name].dropna()
        if len(age_sex_data) < 10:
            age_sex_data = data[
                (data['SEX'] == sex) &
                (data['AGE'] >= max(0, min_age - 10)) &
                (data['AGE'] < min(120, max_age + 10))
            ][test_name].dropna()
        if len(age_sex_data) < 5:
            age_sex_data = data[data['SEX'] == sex][test_name].dropna()
        percentile = (age_sex_data < test_value).sum() / len(age_sex_data) * 100
        return percentile
    except Exception as e:
        raise Exception(f"Error calculating percentile: {str(e)}")

def classify_result_clinical(data, test_name, age, sex, test_value):
    try:
        reference_range = calculate_clinical_reference_range(data, test_name, age, sex)
        percentile = calculate_percentile(data, test_name, age, sex, test_value)
        reference_range['percentile'] = percentile
        if reference_range['lower'] <= test_value <= reference_range['upper']:
            classification = "Normal"
        else:
            classification = "Abnormal"
        return classification, percentile, reference_range
    except Exception as e:
        raise Exception(f"Error classifying result clinically: {str(e)}")

def get_clinical_interpretation(test_value, reference_range):
    try:
        lower = reference_range['lower']
        upper = reference_range['upper']
        std = reference_range['std']
        interpretation = {
            'status': 'Normal',
            'severity': None,
            'description': 'Within normal limits'
        }
        if test_value < lower:
            interpretation['status'] = 'Low'
            if test_value < lower - 2 * std:
                interpretation['severity'] = 'Critically Low'
                interpretation['description'] = 'Significantly below normal range'
            elif test_value < lower - std:
                interpretation['severity'] = 'Moderately Low'
                interpretation['description'] = 'Moderately below normal range'
            else:
                interpretation['severity'] = 'Mildly Low'
                interpretation['description'] = 'Slightly below normal range'
        elif test_value > upper:
            interpretation['status'] = 'High'
            if test_value > upper + 2 * std:
                interpretation['severity'] = 'Critically High'
                interpretation['description'] = 'Significantly above normal range'
            elif test_value > upper + std:
                interpretation['severity'] = 'Moderately High'
                interpretation['description'] = 'Moderately above normal range'
            else:
                interpretation['severity'] = 'Mildly High'
                interpretation['description'] = 'Slightly above normal range'
        return interpretation
    except Exception as e:
        raise Exception(f"Error getting clinical interpretation: {str(e)}")
