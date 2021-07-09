
local mlm_train_data = std.extVar('mlm_train_data');

{
  dataset_reader: {
    type: 'extend_with_mlm',
    tokenizer: {
      type: 'digits_aware',
      transformer_model_name: 'bert-base-uncased',
      include_more_numbers: true,
    },
    shuffle: true,
    max_seq_len: 512,
    mlm_train_data: 'datasets/injecting_numeracy/mlm.jsonl',
    dataset_reader: {
      type: 'my_drop',
      transformer_model_name: 'bert-base-uncased',
      include_more_numbers: true,
      max_instances: 524288,
      shuffle: true,
    },
  },
  train_data_path: 'datasets/injecting_numeracy/injecting_numeracy_all_train.json',
  validation_data_path: 'datasets/injecting_numeracy/injecting_numeracy_all_dev.json',
}
