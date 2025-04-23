#!/bin/bash

for method in   prefix prompt none
do
  python main.py --method $method --epochs 5 
done