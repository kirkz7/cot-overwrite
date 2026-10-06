# Training-data inputs that are git-ignored on main

reason_decoupled_train_first1000.jsonl: the first 1000 rows of the desktop's data_train/reason_decoupled_train.jsonl
(E12 reasoning data, gen_train_data.py + the E12 reasoning targets). E18 / E18.1 mix in exactly these 1000 rows.
Restore as data_train/reason_decoupled_train.jsonl (the generators read only the first 1000 rows):

    mkdir -p data_train
    git show origin/weights-e18:data/reason_decoupled_train_first1000.jsonl > data_train/reason_decoupled_train.jsonl
    sha256sum data_train/reason_decoupled_train.jsonl   # must be 21de579406e02d0dcb1c1a45a154a30d71ef2e2cec0fb4c68e5b93fb944252bf

Check: python gen_bind_data_v3.py must reproduce data_train/bind3_decoupled_train.jsonl with sha256 prefix 4BAFEB760E58B565.
