# GenBERT

AllenNLP wrapper of 
[GenBERT](https://github.com/ag1988/injecting_numeracy).


## Dependencies

The latest version of AllenNLP (maybe version >= 2.0 should be fine).

```sh
pip install allennlp allennlp-models
```

## Dataset preprocessing

Create `datasets` directory and expand there [the DROP dataset](https://allennlp.org/drop.html).

Then, preprocess the dataset json files into pickled ones:

```sh
python -m genbert.datasets.drop config/drop_dataset.jsonnet
```

which process tha files specified in `train_data_path` and `validation_data_path` in the jsonnet file and can be equivalently written as:

```sh
for target in train dev; do
    python -m genbert.datasets.drop config/drop_dataset.jsonnet \
        --input-file datasets/drop_dataset/drop_dataset_${target}.json \
        --output-file datasets/drop_dataset/drop_dataset_${target}.pickle
done
```


## Training

```sh
env \
  seed=42 \
  train_data_path=datasets/drop_dataset/drop_dataset_train.pickle  \
  validation_data_path=datasets/drop_dataset/drop_dataset_dev.pickle  \
  devices="1,2,3,4"  \      # use four devices
$ allennlp train --serilize-dir results --include-package genbert configs/genbert.jsonnet
```