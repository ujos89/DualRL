import pandas as pd
import os

# raw txt testset to 
def read_file(path_file, file_name, label):
    f = open(os.path.join(path_file, file_name))
    texts = (f.read().split('\n')[:-1])
    df = pd.DataFrame(texts, columns=['text'])
    df['label'] = label
    f.close()

    return df

def split_dataset(path_root, extension):
    file_names = os.listdir(path_root)
    label_dict = {'neg':-1, 'negpos':1, 'neu':0, 'neuneg':-1, 'neupos':1, 'pos':1, 'posneg':-1}
    path_save = '../dataset/split'
    
    for file_name in file_names:
        if file_name.endswith(extension):
            file_name_ = file_name.split('.')[0]
            label = label_dict[file_name_]
            df = read_file(path_root, file_name, label)
                        
            df, df_test = train_test_split(df, test_size=.2)
            df_train, df_val = train_test_split(df, test_size=.125)
            
            df_test.to_csv(os.path.join(path_save, file_name_+'.test.csv') ,index=False)
            df_val.to_csv(os.path.join(path_save, file_name_+'.val.csv') ,index=False)
            df_train.to_csv(os.path.join(path_save, file_name_+'.train.csv') ,index=False)