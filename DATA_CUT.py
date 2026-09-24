import pandas as pd

try:
    df = pd.read_csv('mental_health_burnout_prediction_dataset.csv')

    selected_columns = [
        'Age', 
        'Gender',
        'Employment_Status',
        'Work_Hours_Per_Week', 
        'Screen_Time_Hours', 
        'Sleep_Hours', 
        'Sleep_Quality', 
        'Physical_Activity_Hours', 
        'Meditation_Minutes', 
        'Coffee_Cups_Per_Day', 
        'Stress_Level',
        'Chronic_Stress',
        'Burnout_Risk' # Target for model training
    ]

    # dataframe (cut)
    new_df = df[selected_columns]

    output_filename = 'user_selected_features.csv'
    new_df.to_csv(output_filename, index=False)

    print(f"File saved successfully as: {output_filename}")
    print("\nShape:", new_df.shape)
    print("\nMissing Values:")
    print(new_df.isnull().sum())
    print("\nFirst 3 rows:")
    print(new_df.head(3))

except Exception as e:
    print("Error:", e)
    