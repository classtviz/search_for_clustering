#!/bin/bash
#SBATCH --partition fn_short
#SBATCH --nodes 1
#SBATCH --mem 48G
#SBATCH --time 0-10:00:00
#SBATCH --job-name jupyter-lab
#SBATCH --output jupyter-lab-%J.log

# get tunneling info
XDG_RUNTIME_DIR=""
port=$(shuf -i8000-9999 -n1)
node=$(hostname -s)
user=$(whoami)
#cluster=$(hostname -f | awk -F"." '{print $2}')
case "$SLURM_SUBMIT_HOST" in
bigpurple-ln1)
login_node=bigpurple1.nyumc.org
;;
bigpurple-ln2)
login_node=bigpurple2.nyumc.org
;;
bigpurple-ln3)
login_node=bigpurple3.nyumc.org
;;
bigpurple-ln4)
login_node=bigpurple4.nyumc.org
;;
esac
# print tunneling instructions jupyter-log
echo -e "
MacOS or linux terminal command to create your ssh tunnel:
ssh -N -L ${port}:${node}:${port} ${user}@${login_node}

For more info and how to connect from windows,
   see research.computing.yale.edu/jupyter-nb
Here is the MobaXterm info:

Forwarded port:same as remote port
Remote server: ${node}
Remote port: ${port}
SSH server: ${login_node}
SSH login: $user
SSH port: 22

Use a Browser on your local machine to go to:
http://localhost:${port}  (prefix w/ https:// if using password)
"

# load modules or conda environments here
# e.g. farnam:
# module unload python
# module load python/gpu/2.7.15

cd /gpfs/data/ruggleslab/
module purge
module unload python
source activate hc_test
jupyter lab --no-browser --port=${port} --ip=${node}
