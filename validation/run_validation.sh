#!/bin/bash
cd "/Users/cathy/Documents/学习相关/老段课题组/AI_project/Mapping-the-Increasing-Use-of-LLMs-in-Scientific-Papers"
/Users/cathy/miniconda3/envs/llm-detection/bin/python \
  "/Users/cathy/Library/Application Support/com.tencent.mac.marvis/MarvisData/User/oAN1i2YiEe89jVMCbZef2QGssqcQ/workspace/conv_19f26b11ee6_a71b6b509f73/temp/run_validation_full.py" \
  > "/Users/cathy/Documents/学习相关/老段课题组/AI_project/validation_log.txt" 2>&1
echo "DONE" >> "/Users/cathy/Documents/学习相关/老段课题组/AI_project/validation_log.txt"
