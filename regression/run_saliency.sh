#!/bin/bash

# Saliency (Integrated Gradients) analysis for all models

# ConvLSTM models
python example_ig_all_frames.py --config-name CONV_1_1_4
python example_ig_all_frames.py --config-name CONV_1_2_6
python example_ig_all_frames.py --config-name CONV_1_3_3
python example_ig_all_frames.py --config-name CONV_2_1_8
python example_ig_all_frames.py --config-name CONV_2_2_6
python example_ig_all_frames.py --config-name CONV_2_3_3
python example_ig_all_frames.py --config-name CONV_3_1_2
python example_ig_all_frames.py --config-name CONV_3_2_1
python example_ig_all_frames.py --config-name CONV_3_3_1
python example_ig_all_frames.py --config-name CONV_4_1_9
python example_ig_all_frames.py --config-name CONV_4_2_2
python example_ig_all_frames.py --config-name CONV_4_3_3
python example_ig_all_frames.py --config-name CONV_5_1_1
python example_ig_all_frames.py --config-name CONV_5_2_6
python example_ig_all_frames.py --config-name CONV_5_3_2
python example_ig_all_frames.py --config-name CONV_6_1_13
python example_ig_all_frames.py --config-name CONV_6_2_7
python example_ig_all_frames.py --config-name CONV_6_3_5
python example_ig_all_frames.py --config-name CONV_7_1_6
python example_ig_all_frames.py --config-name CONV_7_2_0
python example_ig_all_frames.py --config-name CONV_7_3_0

# Transformer models
python example_ig_all_frames.py --config-name TRANSFORMER_1_1_4
python example_ig_all_frames.py --config-name TRANSFORMER_1_2_6
python example_ig_all_frames.py --config-name TRANSFORMER_1_3_3
python example_ig_all_frames.py --config-name TRANSFORMER_2_1_8
python example_ig_all_frames.py --config-name TRANSFORMER_2_2_6
python example_ig_all_frames.py --config-name TRANSFORMER_2_3_3
python example_ig_all_frames.py --config-name TRANSFORMER_3_1_2
python example_ig_all_frames.py --config-name TRANSFORMER_3_2_1
python example_ig_all_frames.py --config-name TRANSFORMER_3_3_1
python example_ig_all_frames.py --config-name TRANSFORMER_4_1_9
python example_ig_all_frames.py --config-name TRANSFORMER_4_2_2
python example_ig_all_frames.py --config-name TRANSFORMER_4_3_3
python example_ig_all_frames.py --config-name TRANSFORMER_5_1_1
python example_ig_all_frames.py --config-name TRANSFORMER_5_2_6
python example_ig_all_frames.py --config-name TRANSFORMER_5_3_2
python example_ig_all_frames.py --config-name TRANSFORMER_6_1_13
python example_ig_all_frames.py --config-name TRANSFORMER_6_2_7
python example_ig_all_frames.py --config-name TRANSFORMER_6_3_5
python example_ig_all_frames.py --config-name TRANSFORMER_7_1_6
python example_ig_all_frames.py --config-name TRANSFORMER_7_2_0
python example_ig_all_frames.py --config-name TRANSFORMER_7_3_0
