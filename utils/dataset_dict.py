import dill
import pickle
with open("/home/houyz/DS_FedCR/overlap_code/data/data_saved/har_dataset.pkl","rb") as f:
    dict_users=dill.load(f)
print(dict_users)