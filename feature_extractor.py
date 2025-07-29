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
    
    # --- Important: Ensure the target variable 'Dementia Status' is always included ---
    target_variable = 'Dementia Status'
    if target_variable not in selected_features:
        selected_features.append(target_variable)
        print(f"Note: The target variable '{target_variable}' has been automatically included.")

    # Filter out any selected features that don't actually exist in the dataframe
    # This prevents errors if the dictionary is from an older version of the data
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
    output_file = 'input_data.csv'

    # --- Step 2: EDIT THIS DICTIONARY ---
    # Change the value from 1 to 0 for any feature you want to REMOVE.
    # Any feature with a 1 will be KEPT.
    feature_selection = {
        'Age at recruitment': 1,
        'Year of birth': 0,
        "Standard PRS for alzheimer's disease (AD)": 0,
        'Townsend deprivation index at recruitment': 0,
        'Number in household | Instance 0': 0,
        'Has_Heart_attack': 0,
        'Has_Angina': 0,
        'Has_Stroke': 0,
        'Has_High_blood_pressure': 0,
        'Has_Any_Vascular_Heart_Problem': 0,
        'Sleep duration | Instance 0': 0,
        'Number of days/week of vigorous physical activity 10+ minutes | Instance 0': 0,
        'Number of days/week of moderate physical activity 10+ minutes | Instance 0': 0,
        'Mean time to correctly identify matches | Instance 0': 0,
        'Fluid intelligence score | Instance 0': 0,
        'Cholesterol | Instance 0': 0,
        'Triglycerides | Instance 0': 0,
        'C-reactive protein | Instance 0': 0,
        'Glucose | Instance 0': 0,
        'Glycated haemoglobin (HbA1c) | Instance 0': 0,
        'Creatinine | Instance 0': 0,
        'Computed_RSA_ln_ms2_Inst0': 0,
        'Average_HR_bpm_Inst0': 0,
        'Average_RR_ms_Inst0': 0,
        'Pretest_RSA_ln_ms2_Inst0': 0,
        'Activity_RSA_ln_ms2_Inst0': 0,
        'Recovery_RSA_ln_ms2_Inst0': 0,
        'Avg_HeartPeriod_ms_Inst0': 0,
        'Dementia Status': 1, # keep this as 1
        'Sex_Female': 0,
        'Sex_Male': 0,
        'Diabetes_Status_Do not know': 0,
        'Diabetes_Status_No': 0,
        'Diabetes_Status_Prefer not to answer': 0,
        'Diabetes_Status_Yes': 0,
        'Smoking status | Instance 0_Current': 0,
        'Smoking status | Instance 0_Never': 0,
        'Smoking status | Instance 0_Prefer not to answer': 0,
        'Smoking status | Instance 0_Previous': 0,
        'Alcohol drinker status | Instance 0_Current': 0,
        'Alcohol drinker status | Instance 0_Never': 0,
        'Alcohol drinker status | Instance 0_Prefer not to answer': 0,
        'Alcohol drinker status | Instance 0_Previous': 0,
        'Qualifications_A-levels/AS-levels or equivalent': 0,
        'Qualifications_CSEs or equivalent': 0,
        'Qualifications_College or University degree': 0,
        'Qualifications_NVQ or HND or HNC or equivalent': 0,
        'Qualifications_None of the above': 0,
        'Qualifications_O-levels/GCSEs or equivalent': 0,
        'Qualifications_Other professional qualifications eg: nursing, teaching': 0,
        'Qualifications_Prefer not to answer': 0,
        'Hearing_aid_user_No': 0,
        'Hearing_aid_user_Prefer not to answer': 0,
        'Hearing_aid_user_Yes': 0,
        'Hearing_difficulty_No': 0,
        'Hearing_difficulty_Prefer not to answer': 0,
        'Hearing_difficulty_Yes': 0,
        'Wears_glasses_or_contact_lenses_No': 0,
        'Wears_glasses_or_contact_lenses_Prefer not to answer': 0,
        'Wears_glasses_or_contact_lenses_Yes': 0,
        'APOE4_carrier_status_Carrier': 0,
        'APOE4_carrier_status_Non-carrier': 0,
        'APOE4_carrier_status_Not available': 0,
        'Systolic_blood_pressure': 0
    }

    # --- Step 3: Run the function ---
    create_feature_subset(original_file, output_file, feature_selection)

