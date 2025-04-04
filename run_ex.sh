#!/bin/bash

# Set the experiment root and cfg file
iterations=200
it_offset=0
export_data=1
verbose=0
fault_max=11
batch_id="dense_30"
cores=1
default_cfg="custom"

## Generate data for a batch of experiments with root given by ex_id_root
for i in {1..6..1}; do
    ex_id=e${ex_id_root}_${i}
    python run_ex_multic.py --ex_id $ex_id --iterations $iterations --it_offset $it_offset \
        --export_data $export_data --verbose $verbose --fault_max $fault_max --batch_id $batch_id \
        --cores $cores --default_cfg $default_cfg
done
