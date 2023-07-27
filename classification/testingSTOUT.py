import pandas as pd
from STOUT import translate_forward
from tqdm import tqdm

# Function to translate SMILES to IUPAC name
def translate_smiles_to_iupac(smiles):
    return translate_forward(smiles)

# Read the CSV file 
csv_file_path = '../data/IL/Cleaned_Final_Data.csv'
df = pd.read_csv(csv_file_path)

unique_iupac_df = df.drop_duplicates(subset='SMILES').copy()

print("Len:", len(unique_iupac_df))

iupacs = []
for smiles in tqdm(unique_iupac_df['SMILES']):
    # print("Processing ",smiles)
    iupacs.append(translate_forward(smiles))
unique_iupac_df['IUPAC'] = iupacs
# unique_iupac_df['IUPAC'] = unique_iupac_df['SMILES'].apply(translate_smiles_to_iupac)

# Save the DataFrame with unique SMILES and their IUPAC names to a new CSV file
output_file = './unique_iupac.csv'
unique_iupac_df.to_csv(output_file, index=False)

print("Unique SMILES and their IUPAC names have been saved to:", output_file)
