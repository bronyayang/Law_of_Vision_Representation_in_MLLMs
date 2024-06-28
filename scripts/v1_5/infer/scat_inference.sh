#!/bin/bash

# Define the model path
MODEL_PATH="/mnt/bn/bohanzhainas1/shijiay/exp_ckpts/02-28-24_LLaVA_clipdiff_mof_ft"

# List of image files
IMAGES=(
    "/opt/tiger/LLaVA1.5/llava/eval/scat/black_cat.jpg"
    "/opt/tiger/LLaVA1.5/llava/eval/scat/cat_tree.png"
    "/opt/tiger/LLaVA1.5/llava/eval/scat/grass_cat.png"
    "/opt/tiger/LLaVA1.5/llava/eval/scat/orange_cat.jpg"
    "/opt/tiger/LLaVA1.5/llava/eval/scat/shelf_cat.png"
)

# List of queries
QUERIES=(
    "Describe this image."
    "What is the main object in this image?"
    "Is there a dog in the image?"
    "What color is this cat?"
    "What objects are in the background?"
)

# Output file
OUTPUT_FILE="inference_mof.txt"

# Loop over each image
for IMAGE in "${IMAGES[@]}"
do
    for QUERY in "${QUERIES[@]}"
    do
        echo "Processing $IMAGE with query: $QUERY" >> "$OUTPUT_FILE"
        python3 llava/eval/run_llava.py \
            --model-path "$MODEL_PATH" \
            --image-file "$IMAGE" \
            --query "$QUERY" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"
    done
done

echo "Inference completed. Results saved in $OUTPUT_FILE"
