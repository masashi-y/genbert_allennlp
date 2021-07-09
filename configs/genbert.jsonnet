local utils = import 'utils.jsonnet';

local train_data_path = std.extVar('train_data_path');
local validation_data_path = std.extVar('validation_data_path');
local seed = std.parseInt(std.extVar('seed'));
local epochs = 40;

local device_info = utils.devices(std.extVar('devices'));

local pretrained = std.extVar('pretrained_weights');


{
  vocabulary: {
    type: 'from_files',
    directory: 'vocabulary',
    padding_token: '[PAD]',
    oov_token: '[UNK]',
  },
  dataset_reader: {
    type: 'pickled',
  },
  validation_dataset_reader: {
    type: 'pickled',
  },
  train_data_path: train_data_path,
  validation_data_path: validation_data_path,
  model: {
    type: 'genbert',
    bert_model_name_or_config: 'bert-base-uncased',
    max_decoding_steps: 50,
    span_prediction_only: false,
    do_random_shift: true,
    masked_lm_loss_coef: 0.5,
    [if pretrained != '' then 'initializer']: {
      regexes: [
        [
          '.*',
          {
            type: 'pretrained',
            weights_file_path: pretrained,
            parameter_name_overrides: {},
          },
        ],
      ],
    },
  },
  data_loader: {
    batch_sampler: {
      type: 'bucket',
      batch_size: 4,
    },
  },
  trainer: {
    optimizer: {
      type: 'huggingface_adamw',
      weight_decay: 0.01,
      parameter_groups: [
        [['bias', 'LayerNorm\\.weight', 'layer_norm\\.weight'], { weight_decay: 0 }],
      ],
      lr: 3e-5,
      eps: 1e-8,
      correct_bias: true,
    },
    learning_rate_scheduler: {
      type: 'linear_with_warmup',
      warmup_steps: 100,
    },
    tensorboard_writer: {
      should_log_learning_rate: true,
    },
    num_epochs: epochs,
    validation_metric: '+per_instance_f1',
    [if !device_info.use_multi_devices then 'cuda_device']: device_info.device_ids[0],
  },
  [if device_info.use_multi_devices then 'distributed']: {
    cuda_devices: device_info.device_ids,
    num_nodes: 1,
  },
  random_seed: seed,
  numpy_seed: seed,
  pytorch_seed: seed,
}
