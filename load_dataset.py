import pandas as pd
from tqdm import tqdm

def load_dataset(file_path, chunk_size=1000000):
    df = pd.DataFrame()
    for chunk in tqdm(pd.read_csv(file_path, iterator=True, chunksize=chunk_size)):
        df = pd.concat([df, chunk], ignore_index=True)

def balanced_dataset_downsampling(chunks, random_state=123):
    df = pd.DataFrame()
    for chunk in tqdm(chunks):
        chunk.reset_index(drop=True, inplace=True)
        
        # Separate by label
        label_1 = chunk[chunk["label"] == 1]
        label_0 = chunk[chunk["label"] == 0]

        # Downsample label 0 to match the size of label 1
        sampled_label_0 = label_0.sample(n=len(label_1), replace=False, random_state=random_state)
        sampled_chunk = pd.concat([label_1, sampled_label_0], ignore_index=True)

        df = pd.concat([df, sampled_chunk], ignore_index=True)
    
    return df

def separate_labels(file_path, label_column="label", chunk_size=1000000):
    df_label_0 = pd.DataFrame()
    df_label_1 = pd.DataFrame()
    
    for chunk in tqdm(pd.read_csv(file_path, chunksize=chunk_size)):
        # Separate rows with label 0 and label 1
        df_label_0 = pd.concat([df_label_0, chunk[chunk[label_column] == 0]], ignore_index=True)
        df_label_1 = pd.concat([df_label_1, chunk[chunk[label_column] == 1]], ignore_index=True)
    

    return df_label_0, df_label_1

### TODO: Create dp dataset
