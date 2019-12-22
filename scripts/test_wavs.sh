#!/usr/bin/env bash
set -e
this_dir="$( cd "$( dirname "$0" )" && pwd )"
test_dir="$(realpath "${this_dir}/../etc/test")"

# Extract English acoustic model
acoustic_model_name="cmusphinx-en-us-ptm-5.2"
acoustic_model_dir="${test_dir}/${acoustic_model_name}"
if [[ ! -d "${acoustic_model_dir}" ]]; then
    acoustic_model_url="https://github.com/synesthesiam/rhasspy-profiles/releases/download/v1.0-en/${acoustic_model_name}.tar.gz"
    echo "Downloading ${acoustic_model_url}"
    wget -O - "${acoustic_model_url}" | tar -C "${test_dir}" -xzvf -
fi

# Create temporary file
temp_file="$(mktemp)"
function finish {
    rm -rf "${temp_file}"
}

trap finish EXIT

ls -1 "${test_dir}/wav"/*.wav > "${temp_file}"

python3 -m rhasspyasr_pocketsphinx_hermes \
        --stdin-files \
        --acoustic-model "${acoustic_model_dir}" \
        --dictionary "${test_dir}/dictionary.txt" \
        --language-model "${test_dir}/language_model.txt" \
        < "${temp_file}" | \
    python3 "${test_dir}/check_transcriptions.py" \
            "${temp_file}" \
            "${test_dir}/wav"
