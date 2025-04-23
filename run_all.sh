#!/bin/bash

for method in  lora prefix prompt
do
  python main.py --method $method --epochs 7 --batch_size 16 --model_name bert-base-uncased
done