#!/bin/sh
dir=$(find . -name hw*murali*spitzer)
venv_dir=$(find $dir/src -name "deeprl_hw*")

zip -r ${dir}.zip\
  ${dir}/src/README.md\
  ${dir}/src/plot_perf.py\
  ${dir}/src/requirements.txt\
  ${venv_dir}/*.py\
  ${dir}/src/dqn_atari.py\
  ${dir}/src/setup.py\
  ${dir}/src/setup_all.sh\
  ${dir}/space_invaders_videos.zip
