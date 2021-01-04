# See all options
help() {
    python mnist_cli.py --help
}

# Sanity check to run 1 iteration of train, test and val
dry_run () {
    python mnist_cli.py \
        --default_root_dir $HOME/logs/ \
        --data_dir $HOME/data \
        --gpus 1 \
        --fast_dev_run 1
}

# Train up to 98% accuracy on 1 gpu
train_to_98 () {
    python mnist_cli.py \
        --default_root_dir $HOME/logs/ \
        --data_dir $HOME/data \
        --gpus 1 \
        --max_epochs 14
}


# Launch Tensorboard (should be installed together with pytorch-lightning
# installation)
# Step 1 : Launch tensorboard
#    `tensorboard --logdir=$HOME/logs/lightning_logs/ --host $SERVER_IP --port 6006
# Step 2 : Allow SSH forwarding (execute on your laptop)
#    `ssh uname@remote.host -L $LOCAL_PORT:$SERVER_IP:$SERVER_PORT`
# where
#     LOCAL_PORT=6006
#     SERVER_IP= (your server IP)
#     SERVER_PORT=6006

# help
# dry_run
train_to_98


