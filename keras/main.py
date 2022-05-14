import variables as var
from prepare_data import train_val_split
from training import train

train_ds, val_ds = train_val_split()

print(train_ds.element_spec)
print(val_ds.element_spec)

train(train_ds, val_ds)

