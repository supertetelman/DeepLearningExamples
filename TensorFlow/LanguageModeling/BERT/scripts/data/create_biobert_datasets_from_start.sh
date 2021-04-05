#!/bin/bash

# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Set the global location for data, models, etc. in the container
export DLE_DATA_DIR="${DLE_DATA_DIR:-/data}" # Defaul data dir for all DeepLearningExamples to /data
export DLE_RESULTS_DIR="${DLE_RESULTS_DIR:-/results}" # Default the results dir for all DeepLearningExamples to /data in the container
export DLE_MODEL_NAME="${DLE_MODEL_NAME:-bert-tf}" # Set the model name, used for data/results subdirectories

export DLE_WORKSPACE_DIR="${DLE_WORKSPACE_DIR:-/workspace/bert}"
export DLE_SCRIPTS_DIR="${DLE_SCRIPTS_DIR:-${DLE_WORKSPACE_DIR}/scripts}"

export DLE_RAW_DATA_DIR="${DLE_RAW_DATA_DIR:-${DLE_DATA_DIR}/${DLE_MODEL_NAME}/raw_data}"
export DLE_PROCESSED_DATA_DIR="${DLE_RAW_DATA_DIR:-${DLE_DATA_DIR}/${DLE_MODEL_NAME}/processed_data}"

export DLE_PRETRAINED_MODELS_DIR="${DLE_PRETRAINED_MODELS_DIR:-${DLE_RESULTS_DIR}/${DLE_MODEL_NAME}/pretrained}"
export DLE_MODEL_RESULTS_DIR="${DLE_MODEL_RESULTS_DIR:-${DLE_RESULTS_DIR}/${DLE_MODEL_NAME}/models}"

mkdir -p "${DLE_DATA_DIR}" "${DLE_RESULTS_DIR}" "${DLE_RAW_DATA_DIR}" "${DLE_PROCESSED_DATA_DIR}" "${DLE_PRETRAINED_MODELS_DIR}" "${DLE_MODEL_RESULTS_DIR}"


# Download
python3 "${DLE_SCRIPTS_DIR}"/data/bertPrep.py --action download --dataset pubmed_baseline

python3 "${DLE_SCRIPTS_DIR}"/data/bertPrep.py --action download --dataset google_pretrained_weights  # Includes vocab

# Properly format the text files
python3 "${DLE_SCRIPTS_DIR}"/data/bertPrep.py --action text_formatting --dataset pubmed_baseline


# Shard the text files
python3 "${DLE_SCRIPTS_DIR}"/data/bertPrep.py --action sharding --dataset pubmed_baseline

### BERT BASE

## UNCASED

# Create TFRecord files Phase 1
python3 "${DLE_SCRIPTS_DIR}"/data/bertPrep.py --action create_tfrecord_files --dataset pubmed_baseline --max_seq_length 128 \
 --max_predictions_per_seq 20 --vocab_file "${DLE_PRETRAINED_MODELS_DIR}"/download/google_pretrained_weights/uncased_L-12_H-768_A-12/vocab.txt


# Create TFRecord files Phase 2
python3 "${DLE_SCRIPTS_DIR}"/data/bertPrep.py --action create_tfrecord_files --dataset pubmed_baseline --max_seq_length 512 \
 --max_predictions_per_seq 80 --vocab_file "${DLE_PRETRAINED_MODELS_DIR}"/download/google_pretrained_weights/uncased_L-12_H-768_A-12/vocab.txt


## CASED

# Create TFRecord files Phase 1
python3 "${DLE_SCRIPTS_DIR}"/data/bertPrep.py --action create_tfrecord_files --dataset pubmed_baseline --max_seq_length 128 \
 --max_predictions_per_seq 20 --vocab_file "${DLE_PRETRAINED_MODELS_DIR}"/download/google_pretrained_weights/cased_L-12_H-768_A-12/vocab.txt \
 --do_lower_case=0


# Create TFRecord files Phase 2
python3 "${DLE_SCRIPTS_DIR}"/data/bertPrep.py --action create_tfrecord_files --dataset pubmed_baseline --max_seq_length 512 \
 --max_predictions_per_seq 80 --vocab_file "${DLE_PRETRAINED_MODELS_DIR}"/download/google_pretrained_weights/cased_L-12_H-768_A-12/vocab.txt \
 --do_lower_case=0
