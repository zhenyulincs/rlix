export GIT_SSH_COMMAND="ssh -i /workspace/.ssh/id_ed25519 -o IdentitiesOnly=yes"
git clone git@github.com:taoluo/_SchedRL.git SchedRL
cd SchedRL
git submodule update --remote --init --checkout external/ROLL_schedrl
# git submodule update --remote --init --checkout external/ROLL_upstream_main
# git submodule update --remote --init --checkout external/ROLL_multi_lora
# git submodule update --remote --init --checkout external/ROLL_multi_pipeline

