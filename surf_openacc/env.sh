export PGI=/opt/pgi;
export PATH=/opt/pgi/linux86-64/17.10/bin:$PATH;
export MANPATH=$MANPATH:/opt/pgi/linux86-64/17.10/man;
sudo nvidia-smi -pm 1
sudo nvidia-smi -ac 2505,875
