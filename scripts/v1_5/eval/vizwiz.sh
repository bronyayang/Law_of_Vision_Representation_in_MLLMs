python -m llava.eval.model_vqa_loader \
    --model-path liuhaotian/llava-v1.5-7b \
    --question-file /mnt/bn/bohanzhainas1/flamingo_data/data/evaluation/vizwiz/test.json \
    --image-folder /mnt/bn/bohanzhainas1/flamingo_data/data/evaluation/vizwiz/test \
    --answers-file /mnt/bn/bohanzhainas1/shijiay/llava_eval/vizwiz/answers/llava-v1.5-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file /mnt/bn/bohanzhainas1/flamingo_data/data/evaluation/vizwiz/test.json \
    --result-file /mnt/bn/bohanzhainas1/shijiay/llava_eval/vizwiz/answers/llava-v1.5-7b.jsonl \
    --result-upload-file /mnt/bn/bohanzhainas1/shijiay/llava_eval/vizwiz/answers_upload/llava-v1.5-7b.json