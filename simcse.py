import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers import models, losses
from torch.utils.data import DataLoader
import tqdm
import logging
# Define your sentence transformer model using CLS pooling
model_name = "DeepChem/ChemBERTa-77M-MLM"
word_embedding_model = models.Transformer(model_name, max_seq_length=128)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Define a list with sentences (1k - 100k sentences)
train_sentences = []
with open('./guacamol_v1_all.smiles', 'r', encoding='utf8') as fIn:
    for line in tqdm.tqdm(fIn):
        line = line.strip()
        if len(line) >= 10:
            train_sentences.append(line)

logging.info("{} train sentences".format(len(train_sentences)))

# Convert train sentences to sentence pairs
train_data = [InputExample(texts=[s, s]) for s in train_sentences]

# DataLoader to batch your data
train_dataloader = DataLoader(train_data, batch_size=256, shuffle=True)

# Use the denoising auto-encoder loss
train_loss = losses.MultipleNegativesRankingLoss(model)

# Call the fit method
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=10,
    show_progress_bar=True
)

model.save('output/simcsesqrt-model')
