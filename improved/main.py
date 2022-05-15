import variables as var
from prepare_data import train_val_split, map_fn
from training import train

# train_ds, val_ds = train_val_split()

# print(train_ds.element_spec)
# print(val_ds.element_spec)

# train(train_ds, val_ds)

(rays_flat, t_vals) = map_fn(var.poses[0])