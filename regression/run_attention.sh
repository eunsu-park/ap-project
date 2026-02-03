#!/bin/bash

# Attention analysis for Transformer models only
# Note: ConvLSTM models don't have attention weights (will be skipped automatically)

# Transformer models
python example_attention_all_targets.py --config-name TRANSFORMER_1_1_4
python example_attention_all_targets.py --config-name TRANSFORMER_1_2_6
python example_attention_all_targets.py --config-name TRANSFORMER_1_3_3
python example_attention_all_targets.py --config-name TRANSFORMER_2_1_8
python example_attention_all_targets.py --config-name TRANSFORMER_2_2_6
python example_attention_all_targets.py --config-name TRANSFORMER_2_3_3
python example_attention_all_targets.py --config-name TRANSFORMER_3_1_2
python example_attention_all_targets.py --config-name TRANSFORMER_3_2_1
python example_attention_all_targets.py --config-name TRANSFORMER_3_3_1
python example_attention_all_targets.py --config-name TRANSFORMER_4_1_9
python example_attention_all_targets.py --config-name TRANSFORMER_4_2_2
python example_attention_all_targets.py --config-name TRANSFORMER_4_3_3
python example_attention_all_targets.py --config-name TRANSFORMER_5_1_1
python example_attention_all_targets.py --config-name TRANSFORMER_5_2_6
python example_attention_all_targets.py --config-name TRANSFORMER_5_3_2
python example_attention_all_targets.py --config-name TRANSFORMER_6_1_13
python example_attention_all_targets.py --config-name TRANSFORMER_6_2_7
python example_attention_all_targets.py --config-name TRANSFORMER_6_3_5
python example_attention_all_targets.py --config-name TRANSFORMER_7_1_6
python example_attention_all_targets.py --config-name TRANSFORMER_7_2_0
python example_attention_all_targets.py --config-name TRANSFORMER_7_3_0
