import pickle
import gzip
import os

models_dir = r'C:\Temp\Classification Models\models'

# Compress all .pkl files
for filename in os.listdir(models_dir):
    if filename.endswith('.pkl'):
        input_path = os.path.join(models_dir, filename)
        output_path = input_path.replace('.pkl', '.pkl.gz')
        
        with open(input_path, 'rb') as f_in:
            with gzip.open(output_path, 'wb') as f_out:
                f_out.writelines(f_in)
        
        print(f"Compressed: {filename}")