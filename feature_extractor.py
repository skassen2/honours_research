import pandas as pd

def create_feature_subset(original_data_path, output_data_path, feature_selection):
    """
    Selects features from a dataset based on a dictionary and saves the result.

    Args:
        original_data_path (str): The path to the input CSV file.
        output_data_path (str): The path to save the new CSV file.
        feature_selection (dict): A dictionary with feature names as keys and 1 (include) or 0 (exclude) as values.
    """
    try:
        # Load the original dataset
        df = pd.read_csv(original_data_path, low_memory=False)
        print(f"Successfully loaded '{original_data_path}' with {df.shape[1]} columns.")
    except FileNotFoundError:
        print(f"Error: The file was not found at '{original_data_path}'")
        print("Please make sure the script is in the same directory as your dataset or update the path.")
        return

    # Identify which columns to keep based on the dictionary
    selected_features = [feature for feature, keep in feature_selection.items() if keep == 1]
    
    # --- Diagnostic check for missing features ---
    all_df_columns = set(df.columns)
    not_found_features = [feature for feature in selected_features if feature not in all_df_columns]

    if not_found_features:
        print("\n⚠️ Warning: The following selected features were not found in the dataset and will be ignored:")
        for feature in not_found_features:
            print(f"- '{feature}'")
        print("Please check for typos or differences between the dictionary keys and the CSV column headers.")
    # ----------------------------------------------------

    # --- Important: Ensure the target variable 'Dementia Status' is always included ---
    target_variable = 'Dementia Status'
    if target_variable not in selected_features and target_variable in df.columns:
        selected_features.append(target_variable)
        print(f"\nNote: The target variable '{target_variable}' has been automatically included.")

    # Filter out any selected features that don't actually exist in the dataframe
    final_columns = [col for col in selected_features if col in df.columns]
    
    # Create the new dataframe with the selected columns
    new_df = df[final_columns]
    
    # Save the new dataframe to a CSV file
    new_df.to_csv(output_data_path, index=False)
    
    print(f"\nSuccessfully created '{output_data_path}' with {new_df.shape[1]} selected features.")
    print("Selected features are:")
    for col in new_df.columns:
        print(f"- {col}")

# ===================================================================
# Main Execution Block
# ===================================================================

if __name__ == '__main__':
    # --- Step 1: Define your input and output file paths ---
    original_file = 'uk_biobank_dataset.csv'
    output_file = 'input_moe_added.csv'

    # --- Step 2: EDIT THIS DICTIONARY ---
    # This is the final, corrected dictionary based on the exact feature names from your file.
    # 1 = KEEP the feature, 0 = REMOVE the feature.
    feature_selection = {
        # --- Demographics & Socioeconomic (Non-Invasive) ---
        'Age at recruitment': 1,
        'Year of birth': 1,
        'Townsend deprivation index at recruitment': 1,
        'Number in household | Instance 0': 0,
        'Sex_Female': 0,
        'Sex_Male': 1,

        # --- Self-Reported Medical History (Non-Invasive) --- CHECK
        'Has_Heart_attack': 0,
        'Has_Angina': 0,
        'Has_Stroke': 0,
        'Has_High_blood_pressure': 0,
        'Has_Any_Vascular_Heart_Problem': 0,
        'Diabetes_Status_Do not know': 0,
        'Diabetes_Status_No': 0,
        'Diabetes_Status_Prefer not to answer': 0,
        'Diabetes_Status_Yes': 0,
        
        # --- Lifestyle & Sensory (Non-Invasive) ---
        'Sleep duration | Instance 0': 0,
        'Number of days/week of vigorous physical activity 10+ minutes | Instance 0': 0,
        'Number of days/week of moderate physical activity 10+ minutes | Instance 0': 0,
        'Smoking status | Instance 0_Current': 0,
        'Smoking status | Instance 0_Never': 0,
        'Smoking status | Instance 0_Prefer not to answer': 0,
        'Smoking status | Instance 0_Previous': 1,
        'Alcohol drinker status | Instance 0_Current': 0,
        'Alcohol drinker status | Instance 0_Never': 0,
        'Alcohol drinker status | Instance 0_Prefer not to answer': 0,
        'Alcohol drinker status | Instance 0_Previous': 1,
        'Alcohol intake frequency. | Instance 0_Daily or almost daily': 0,
        'Alcohol intake frequency. | Instance 0_Never': 0,
        'Alcohol intake frequency. | Instance 0_Once or twice a week': 0,
        'Alcohol intake frequency. | Instance 0_One to three times a month': 1,
        'Alcohol intake frequency. | Instance 0_Prefer not to answer': 0,
        'Alcohol intake frequency. | Instance 0_Special occasions only': 0,
        'Alcohol intake frequency. | Instance 0_Three or four times a week': 0,
        'Bipolar and major depression status | Instance 0_Bipolar I Disorder': 0,
        'Bipolar and major depression status | Instance 0_Bipolar II Disorder': 0,
        'Bipolar and major depression status | Instance 0_No Bipolar or Depression': 0,
        'Bipolar and major depression status | Instance 0_Probable Recurrent major depression (moderate)': 1,
        'Bipolar and major depression status | Instance 0_Probable Recurrent major depression (severe)': 1,
        'Bipolar and major depression status | Instance 0_Single Probable major depression episode': 0,
        'Probable recurrent major depression (moderate) | Instance 0_Yes': 0,
        'Probable recurrent major depression (severe) | Instance 0_Yes': 0,
        'Single episode of probable major depression | Instance 0_Yes': 0,
        'Worrier / anxious feelings | Instance 0_Do not know': 0,
        'Worrier / anxious feelings | Instance 0_No': 0,
        'Worrier / anxious feelings | Instance 0_Prefer not to answer': 0,
        'Worrier / anxious feelings | Instance 0_Yes': 0,

        # --- Cognitive & Physical Measurements (Non-Invasive) --- #CHECK
        'Mean time to correctly identify matches | Instance 0': 0,
        'Fluid intelligence score | Instance 0': 0,
        'Computed_RSA_ln_ms2_Inst0': 0,
        'Average_HR_bpm_Inst0': 0,
        'Average_RR_ms_Inst0': 0,
        'Pretest_RSA_ln_ms2_Inst0': 0,
        'Activity_RSA_ln_ms2_Inst0': 0,
        'Recovery_RSA_ln_ms2_Inst0': 0,
        'Avg_HeartPeriod_ms_Inst0': 0,
        'Hand grip strength | Instance 0': 1,
        'Gate_speed_slow': 1,
        # Note: 'Systolic_blood_pressure' was not in your final list, so it has been removed.

        # --- Invasive Features (EXCLUDED) ---
        "Standard PRS for alzheimer's disease (AD)": 0, # Genetic test
        'Cholesterol | Instance 0': 0, # Blood test
        'Triglycerides | Instance 0': 0, # Blood test
        'C-reactive protein | Instance 0': 0, # Blood test
        'Glucose | Instance 0': 0, # Blood test
        'Glycated haemoglobin (HbA1c) | Instance 0': 0, # Blood test
        'Creatinine | Instance 0': 0, # Blood test
        
        # --- Target Variable (Always Included) ---
        'Dementia Status': 1
    }

    # --- Step 3: Run the function ---
    create_feature_subset(original_file, output_file, feature_selection)

