#!/bin/bash
#SBATCH --job-name=c3e_test_lmhf
#SBATCH --partition=test
#SBATCH --gres=gpu:a100:1
#SBATCH --nodelist=gnode009
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:20:00
#SBATCH --output=/data/wesleyferreiramaia/wokzone-alpamayo/logs/c3e_test_lmhf_%j.out
#SBATCH --error=/data/wesleyferreiramaia/wokzone-alpamayo/logs/c3e_test_lmhf_%j.err

BASE=/data/wesleyferreiramaia/wokzone-alpamayo
CF=$BASE/cosmos-framework

export HF_HOME=$BASE/.hf_cache
cd $CF
.venv/bin/python $BASE/mi3lab-workzone-vla/diagnostics/test_cosmos3edge_lm_hf.py
