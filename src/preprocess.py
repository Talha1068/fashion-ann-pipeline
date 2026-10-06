import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

p = yaml.safe_load(open("params.yaml"))["preprocess"]
d = np.load("data/raw/fashion_mnist.npz")


def normalize(x):
    # Reconciled: teammate used standardization, main used [-1, 1] scaling.
    # Keep the [0, 1] scaling required by the assignment.
    return x.astype("float32") / 255.0


x_train = normalize(d["x_train"])
x_test = normalize(d["x_test"])
x_train, x_val, y_train, y_val = train_test_split(
    x_train, d["y_train"], test_size=p["test_size"],
    random_state=p["seed"], stratify=d["y_train"])

os.makedirs("data/processed", exist_ok=True)
np.savez_compressed("data/processed/data.npz", x_train=x_train, y_train=y_train,
                    x_val=x_val, y_val=y_val, x_test=x_test, y_test=d["y_test"])
print("Saved processed data")