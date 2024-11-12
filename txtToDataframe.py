import json
import re

# Path to the input and output files
input_file = 'Training.txt'
output_file = 'Training.json'

# Read the input file
with open(input_file, 'r', encoding='ISO-8859-1') as file:
    # Read the first line to get the column headers
    headers = file.readline().strip().split('\t')
    
    # Initialize an empty list to store the data
    data = []

    # Process the remaining lines
    for line in file:
        # Split each line by tabs
        row = line.strip().split('\t')
        
        if 2 < len(row):

            row[2] = re.sub(r'@.*? ','', row[2])

        # Create a dictionary by zipping headers and row values
        entry = dict(zip(headers, row))
        
        # Append the entry to the data list
        data.append(entry)

# Write the data to a JSON file
with open(output_file, 'w', encoding='utf-8') as json_file:
    json.dump(data, json_file, indent=4)

print(f"Data has been successfully converted to {output_file}.")