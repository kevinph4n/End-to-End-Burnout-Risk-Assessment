import pandas as pd

# Load raw dataset
try:
    df = pd.read_csv('mental_health_burnout_prediction_dataset.csv')
    
    # Define target columns
    target_columns = [
        'Age', 'Gender', 'Employment_Status', 'Work_Hours_Per_Week', 
        'Screen_Time_Hours', 'Sleep_Hours', 'Sleep_Quality', 
        'Physical_Activity_Hours', 'Meditation_Minutes', 
        'Coffee_Cups_Per_Day', 'Stress_Level', 'Chronic_Stress',
        'Burnout_Risk', 'Burnout_Score', 'Productivity_Score', 'Mental_Health_Status'
    ]
    
    # Filter dataframe (handling missing columns if any)
    existing_columns = [col for col in target_columns if col in df.columns]
    df_filtered = df[existing_columns]
    
    # Save to new csv
    output_filename = 'filtered_dataset.csv'
    df_filtered.to_csv(output_filename, index=False)
    
    print(f"Successfully created {output_filename} with columns:")
    print(existing_columns)
    print(f"\nShape: {df_filtered.shape}")
except Exception as e:
    print(f"Error: {e}")