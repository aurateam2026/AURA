mixed_4_0_w_sys+terry_1qNa_v1_1_15p
从v1.2开始检测，v1.2是验证后的数据，表示从中抽取了15%的数据进行训练


qwen3vl-8b_20260304_02
```python 
# 1qna，让模型学习应该什么时候结束，后33%的数据截断，让模型除了在视频结束的时候停止回答，还需要在视频中间的时候停止回答
qwen3vl-8b/full_silent=balance/sft_Gemini_data_v2_1_basic_obj_ch_4_5_w_sys+basic_obj_en_4_5_w_sys+basic_sub_ch_3_0_w_sys+basic_sub_en_3_0_w_sys+proactive_4_0_keep_45k_w_sys+mixed_4_0_w_sys+terry_1qNa_v1_2_15p_truncated33_w_sys_1e-5_ep1_new_loss
# 从03里面找
```




1. qwen3vl-8b_20260310_01

```python 
# 使用了ovobench的刷榜数据进行训练
qwen3vl-8b/full_silent=balance/sft_Gemini_data_v2_1_basic_obj_ch_4_5_w_sys+basic_obj_en_4_5_w_sys+basic_sub_ch_3_0_w_sys+basic_sub_en_3_0_w_sys+proactive_4_0_keep_45k_w_sys+mixed_4_0_w_sys+ovobench_v1_1e-5_ep1/checkpoint-845
# 和qwen3vl-8b_20260225_02比较，比较ovobench刷榜数据的影响
qwen3vl-8b/full_silent=balance/sft_Gemini_data_v2_1_basic_obj_ch_4_5_w_sys+basic_obj_en_4_5_w_sys+basic_sub_ch_3_0_w_sys+basic_sub_en_3_0_w_sys+proactive_4_0_keep_45k_w_sys+mixed_4_0_w_sys_1e-5_ep1/
```


2. qwen3vl-8b_20260311_01
```python 
# 使用了ovobench的刷榜数据进行训练
# 并切分input，比如切分成1s一个chunk作为输入
# chunkwise
qwen3vl-8b/full_silent=balance/sft_Gemini_data_v2_1_basic_obj_ch_4_5_w_sys+basic_obj_en_4_5_w_sys+basic_sub_ch_3_0_w_sys+basic_sub_en_3_0_w_sys+proactive_4_0_keep_45k_w_sys+mixed_4_0_w_sys+ovobench_v1_chunkwise_w_sys_1e-5_ep1/checkpoint-845
```


3. qwen3vl-8b_20260311_02/
```python 
# terry_1qNa_v2，表示质检后的1qna数据，抽取了35k（=15% 1qna）条数据 
qwen3vl-8b/full_silent=balance/sft_Gemini_data_v2_1_basic_obj_ch_4_5_w_sys+basic_obj_en_4_5_w_sys+basic_sub_ch_3_0_w_sys+basic_sub_en_3_0_w_sys+proactive_4_0_keep_45k_w_sys+mixed_4_0_w_sys+terry_1qNa_v2_35k_w_sys_1e-5_ep1/
# 和qwen3vl-8b_20260225_02比较，比较质检后的1qna数据的影响
```


4. qwen3vl-8b_20260311_03/
```python 
# 4.5分，1w+条数据
qwen3vl-8b/full_silent=balance/sft_Gemini_data_v2_1_basic_obj_ch_4_5_w_sys+basic_obj_en_4_5_w_sys+basic_sub_ch_3_0_w_sys+basic_sub_en_3_0_w_sys+proactive_4_0_keep_45k_w_sys+mixed_4_0_w_sys+terry_1qNa_v2_4_5_w_sys_1e-5_ep1/
```

<!-- 
5. qwen3vl-8b_20260311_04/
```python   
# 使用了office的刷榜数据进行训练, 50条左右数据
qwen3vl-8b/full_silent=balance/sft_Gemini_data_v2_1_basic_obj_ch_4_5_w_sys+basic_obj_en_4_5_w_sys+basic_sub_ch_3_0_w_sys+basic_sub_en_3_0_w_sys+proactive_4_0_keep_45k_w_sys+mixed_4_0_w_sys+terry_1qNa_v2_4_5_w_sys+office_batch1_w_sys_1e-5_ep1/
``` -->

