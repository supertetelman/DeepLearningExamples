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
export DLE_MODEL_NAME="${DLE_MODEL_NAME:-bert}" # Set the model name, used for data/results subdirectories # TODO: Consider renaming to bert-tf1

export DLE_WORKSPACE_DIR="${DLE_WORKSPACE_DIR:-/workspace/bert}"
export DLE_SCRIPTS_DIR="${DLE_SCRIPTS_DIR:-${DLE_WORKSPACE_DIR}/scripts}"

export DLE_RAW_DATA_DIR="${DLE_RAW_DATA_DIR:-${DLE_DATA_DIR}/${DLE_MODEL_NAME}/raw_data}"
export DLE_PROCESSED_DATA_DIR="${DLE_PROCESSED_DATA_DIR:-${DLE_DATA_DIR}/${DLE_MODEL_NAME}/processed_data}"

export DLE_PRETRAINED_MODELS_DIR="${DLE_PRETRAINED_MODELS_DIR:-${DLE_RESULTS_DIR}/${DLE_MODEL_NAME}/pretrained}"
export DLE_MODEL_RESULTS_DIR="${DLE_MODEL_RESULTS_DIR:-${DLE_RESULTS_DIR}/${DLE_MODEL_NAME}/models}"

mkdir -p "${DLE_DATA_DIR}" "${DLE_RESULTS_DIR}" "${DLE_RAW_DATA_DIR}" "${DLE_PROCESSED_DATA_DIR}" "${DLE_PRETRAINED_MODELS_DIR}" "${DLE_MODEL_RESULTS_DIR}"


function download_data() {
  #Download
  if [ "$to_download" = "wiki_books" ] ; then
    python3 "${DLE_SCRIPTS_DIR}"/data/bertPrep.py --action download --dataset bookscorpus
  fi

  python3 "${DLE_SCRIPTS_DIR}"/data/bertPrep.py --action download --dataset wikicorpus_en
  python3 "${DLE_SCRIPTS_DIR}"/data/bertPrep.py --action download --dataset squad
  python3 "${DLE_SCRIPTS_DIR}"/data/bertPrep.py --action download --dataset mrpc
  python3 "${DLE_SCRIPTS_DIR}"/data/bertPrep.py --action download --dataset sst-2
  python3 ${BERT_PREP_WORKING_DIR}/bertPrep.py --action download --dataset google_pretrained_weights # TODO
}


function download_pretrained() {
  #SQuAD Large Checkpoint
  echo "Downloading SQuAD Large Checkpoint"
  cd "${DLE_PRETRAINED_MODELS_DIR}" && \
    wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/bert_tf_ckpt_large_qa_squad11_amp_384/versions/19.03.1/zip -O bert_tf_ckpt_large_qa_squad11_amp_384_19.03.1.zip && \
    unzip bert_tf_ckpt_large_qa_squad11_amp_384_19.03.1.zip -d bert_tf_squad11_large_384 && \
    rm bert_tf_ckpt_large_qa_squad11_amp_384_19.03.1.zip

  #SQuAD Base Checkpoint
  cd "${DLE_SCRIPTS_DIR}"/data/download/nvidia_pretrained && \
    wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/bert_tf_ckpt_base_qa_squad11_amp_128/versions/19.03.1/zip -O bert_tf_ckpt_base_qa_squad11_amp_128_19.03.1.zip && \
    unzip bert_tf_ckpt_base_qa_squad11_amp_128_19.03.1.zip -d bert_tf_squad11_base_128 && \
    rm bert_tf_ckpt_base_qa_squad11_amp_128_19.03.1.zip

  #Pretraining Large checkpoint
  cd /workspace/bert/data/download/nvidia_pretrained && \
    wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/bert_tf_ckpt_large_pretraining_amp_lamb/versions/19.03.1/zip -O bert_tf_ckpt_large_pretraining_amp_lamb_19.03.1.zip && \
    unzip bert_tf_ckpt_large_pretraining_amp_lamb_19.03.1.zip -d bert_tf_pretraining_large_lamb && \
    rm bert_tf_ckpt_large_pretraining_amp_lamb_19.03.1.zip

  python3 "${DLE_SCRIPTS_DIR}"/data/bertPrep.py --action download --dataset google_pretrained_weights  # Redundant, to verify and remove
}


function process_data() {
  DATASET="wikicorpus_en"
  # Properly format the text files
  if [ "$to_download" = "wiki_books" ] ; then
    python3 "${DLE_SCRIPTS_DIR}"/data/bertPrep.py --action text_formatting --dataset bookscorpus
    DATASET="books_wiki_en_corpus"
  fi
  python3 "${DLE_SCRIPTS_DIR}"/data/bertPrep.py --action text_formatting --dataset wikicorpus_en

  # Shard the text files
  python3 "${DLE_SCRIPTS_DIR}"/data/bertPrep.py --action sharding --dataset $DATASET
}
  

function create_tf_record() {
  # Create TFRecord files Phase 1
  # TODO
  python3 "${DLE_SCRIPTS_DIR}"/data/bertPrep.py --action create_tfrecord_files --dataset ${DATASET} --max_seq_length 128 \
 --max_predictions_per_seq 20 --vocab_file ${BERT_PREP_WORKING_DIR}/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt

  # Create TFRecord files Phase 2
  # TODO
  python3 "${DLE_SCRIPTS_DIR}"/data/bertPrep.py --action create_tfrecord_files --dataset ${DATASET} --max_seq_length 512 \
 --max_predictions_per_seq 80 --vocab_file ${BERT_PREP_WORKING_DIR}/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt
}


function main() {
  to_download=${1:-"wiki_only"} # By default, we don't download BooksCorpus dataset due to recent issues with the host server

  download_data ${@}
  download_pretrained ${@}
  process_data ${@}
  create_tf_record ${@}
}


main ${@}
