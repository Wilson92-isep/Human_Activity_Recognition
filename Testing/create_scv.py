import scipy.io
import csv
import os

def mat_files_to_csv(mat_folders, csv_file):
    # Create an empty dictionary to store the data
    data = {"sensor" : [], "activity" : []}
    
    # Iterate over each folder
    for mat_folder in mat_folders:

        mat_folder = os.path.join("../../../USC-HAD",mat_folder)
        # Get a list of all MATLAB files in the folder
        mat_files = [f for f in os.listdir(mat_folder) if f.endswith('.mat')]

        # Iterate over each MATLAB file
        for mat_file in mat_files:
            # Load MATLAB data from the .mat file
            file_path = os.path.join(mat_folder, mat_file)
            mat_data = scipy.io.loadmat(file_path)
            
            # Extract the variable names from the data
            variables = mat_data.keys()
            
            # Add the variable values to the data dictionary
            try :   
                data['sensor'].append(mat_data["sensor_readings"])
                data['activity'].append(mat_data['activity_number'])
            except :
                print("cancel")

    
    
    
        # Open the CSV file in write mode

    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the variable names as the CSV header
        writer.writerow(data.keys())
        
        # Write the variable values as rows in the CSV file
        for i in range(len(data['sensor'])-1):
            row = [data[variable][i] for variable in data.keys()]
            writer.writerow(row)
    
    print(f"Conversion complete. CSV file saved as {csv_file}")

# Usage example
mat_folders = ['Subject1', 'Subject10', 'Subject11', 'Subject12', 'Subject13', 'Subject14', 'Subject2', 'Subject3', 'Subject4', 'Subject5', 'Subject6', 'Subject7', 'Subject8', 'Subject9'] # Replace with the paths to your folders containing MATLAB files
csv_file = "merged_data.csv"  # Replace with the desired path for the merged CSV file

mat_files_to_csv(mat_folders, csv_file)

