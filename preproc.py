# Imports
import os
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.model_selection import train_test_split
# Tensorflow/Keras Imports
os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"
import tensorflow as tf
import tensorflow_hub as hub
import keras
# Restrict TensorFlow to only use the /GPU:0
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)
# Set default Tensor type to double (float64)
keras.backend.set_floatx('float64')

# Ensure dump and parquets folders are ready
if not os.path.isdir("./dump"):
    os.makedirs("./dump")
if not os.path.isdir("./parquets"):
    os.makedirs("./parquets")

# Prepare the Universal Sentence Encoder (USE)
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Function to split 70% Training vs 10% Testing vs 20% Validation
def train_test_vali(df:pd.DataFrame):
    trSz:float = 0.3 # Leftover from training (1 - 0.7 = 0.3)
    teSz:float = 0.1 # Testing ratio

    # Calculate the training portion and the remainder
    tra, rem = train_test_split( # No need to split into X and Y
        df, test_size=trSz, random_state=42
    )
    # Use the remainder to calculate the testing and validation portions
    val, tes = train_test_split(
        rem, test_size=float(teSz/trSz), random_state=42
    )
    # Return the entire tuple
    return (tra, tes, val) # Still DataFrames

# Import dataset
sourceDF:pd.DataFrame = pd.read_csv(
    "./readOnly/database.csv", index_col=1
).drop(columns=["Unnamed: 0"]) # Drop unused column

# Basic statistics
print("sourceDF:")
print(f"Shape: {sourceDF.shape}")
print(f"Data Types:\n{sourceDF.dtypes}")
# Check for number of NaNs
print(f"NaNs Count:\n{sourceDF.isnull().sum()}\n\n")

# Clean Data (NaNs for Abstracts) (only "abstract" has some NaNs)
# Drop papers w NaN values
cleanedDF = sourceDF.dropna(
    axis=0, subset=["abstract"]
).reset_index(drop=True)

# Extract the titles and abstracts
titles:list[str] = list(cleanedDF["title"])
authors:list[str] = list(cleanedDF["authors"])
abstracts:list[str] = list(cleanedDF["abstract"])

# Vectorize via the Universal Sentence Encoder (USE)
titleEmbeddings:npt.NDArray = np.array(embed(titles).numpy())
authorEmbeddings:npt.NDArray = np.array(embed(authors).numpy())
abstractEmbeddings:npt.NDArray = np.array(embed(abstracts).numpy())
titleEmbDF:pd.DataFrame = pd.DataFrame(
    titleEmbeddings,
    columns=[
        f"title_emb_{i}"
        for i in range(titleEmbeddings.shape[1])
    ]
)
authorEmbDF:pd.DataFrame = pd.DataFrame(
    authorEmbeddings,
    columns=[
        f"authors_emb_{i}"
        for i in range(authorEmbeddings.shape[1])
    ]
)
abstractEmbDF:pd.DataFrame = pd.DataFrame(
    abstractEmbeddings,
    columns=[
        f"abstract_emb_{i}"
        for i in range(abstractEmbeddings.shape[1])
    ]
)

# Construct the vectorized dataset
# Not using links in predictions
vectorizedDF:pd.DataFrame = pd.concat(
    [
        pd.DataFrame({
            "year": cleanedDF["year"],
            "citations": cleanedDF["citations"],
        }).reset_index(drop=True),
        titleEmbDF.reset_index(drop=True),
        # authorEmbDF.reset_index(drop=True),
        abstractEmbDF.reset_index(drop=True)
    ],
    axis=1
).astype({ # Ensure these ones are np.int64
    "year": np.int64,
    "citations": np.int64,
})

# Training/Testing/Validation Splitting
(
    trnDF, tstDF, valDF
) = train_test_vali(vectorizedDF)
print(trnDF.shape, tstDF.shape, valDF.shape)

# Store as parquets for later
cleanedDF.to_parquet(
    "parquets/cleaned.parquet.gzip",
    compression="gzip"
)
vectorizedDF.to_parquet(
    "parquets/vectorized.parquet.gzip",
    compression="gzip"
)
trnDF.to_parquet(
    "parquets/train.parquet.gzip",
    compression="gzip"
)
tstDF.to_parquet(
    "parquets/test.parquet.gzip",
    compression="gzip"
)
valDF.to_parquet(
    "parquets/val.parquet.gzip",
    compression="gzip"
)