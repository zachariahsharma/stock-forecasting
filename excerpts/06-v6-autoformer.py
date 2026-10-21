# Source: notebooks/TrainStrengthArchitecture.ipynb · cell 66
# Archived Hugging Face configuration; experimental implementation.
# Viewing excerpt; see original notebook for context.

from transformers import AutoformerConfig, AutoformerForPrediction
configuration = AutoformerConfig(
    prediction_length=1,
    context_length=len(lags_sequence),
    input_size=5,
    lags_sequence=lags_sequence,
    num_time_features=len(time_features) + 1,
    num_static_categorical_features=1,
    cardinality=[len(dataset)],
    embedding_dimension=[20],
    encoder_layers=4,
    decoder_layers=4,
    d_model=32,
)
model = AutoformerForPrediction(configuration)
