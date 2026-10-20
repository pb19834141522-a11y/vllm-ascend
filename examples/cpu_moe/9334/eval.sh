ais_bench \
  --work-dir /mnt/share/y50063808/vllm-ascend-workspace/outputs \
  --models 8150 \
  --datasets gsm8k_gen_0_shot_cot_str_perf \
  --summarizer default_perf \
  --mode perf \
  --num-prompts 2 \
  --num-warmups 1



ais_bench \
  --work-dir /mnt/share/y50063808/vllm-ascend-workspace/outputs \
  --models 8150 \
  --datasets gsm8k_gen_0_shot_cot_chat_prompt \
  --mode all \
  --dump-eval-details \
  --merge-ds \
  --num-warmups 0

    # --num-prompts 200 \
