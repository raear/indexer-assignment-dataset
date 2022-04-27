import math
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


class Dataset:

    def __init__(self, data, batch_size, cache_dir, max_length=512):
        self.data = data
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/specter", cache_dir=cache_dir)

    def __len__(self):
        return len(self.data)

    def batches(self):
        batch = []
        batch_ids = []
        batch_size = self.batch_size
        i = 0
        for k, d in self.data.items():
            if (i) % batch_size != 0 or i == 0:
                batch_ids.append(k)
                batch.append(d["title"] + self.tokenizer.sep_token + (d.get("abstract") or ""))
            else:
                input_ids = self.tokenizer(batch, padding=True, truncation=True, 
                                           return_tensors="pt", max_length=self.max_length)
                yield input_ids.to("cuda"), batch_ids
                batch_ids = [k]
                batch = [d["title"] + self.tokenizer.sep_token + (d.get("abstract") or "")]
            i += 1
        if len(batch) > 0:
            input_ids = self.tokenizer(batch, padding=True, truncation=True, 
                                       return_tensors="pt", max_length=self.max_length)        
            input_ids = input_ids.to("cuda")
            yield input_ids, batch_ids


class SpecterModel:
    def __init__(self, cache_dir):
        self.model = AutoModel.from_pretrained("allenai/specter", cache_dir=cache_dir)
        self.model.to("cuda")
        self.model.eval()

    def __call__(self, input_ids):
        output = self.model(**input_ids)
        return output.last_hidden_state[:, 0, :] # cls token


def create_embeddings(data, batch_size, cache_dir):
    dataset = Dataset(data=data, batch_size=batch_size, cache_dir=cache_dir)
    model = SpecterModel(cache_dir=cache_dir)
    
    id_list = []
    embedding_list = []
    for batch, batch_ids in tqdm(dataset.batches(), total=math.ceil(len(dataset)/dataset.batch_size)):
        pred = model(batch)
        for _id, embedding in zip(batch_ids, pred.unbind()):
            id_list.append(_id)
            embedding_list.append(embedding.detach().cpu().numpy().reshape([1,-1]))

    ids = np.array(id_list, dtype=np.int).reshape([-1,1])
    embeddings = np.concatenate(embedding_list, axis=0)
    return ids, embeddings