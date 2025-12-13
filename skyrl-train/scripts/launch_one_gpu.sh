#!/bin/bash
srun --nodes=1 --ntasks=1 --cpus-per-task=20 --mem=200G --gres=gpu:1 -t 1:00:00 -p ailab --pty bash