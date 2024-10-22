# EE798R_Paper_1_Implementation

First we work on the stable diffusion guided model.

cd D:\Courses\7th sem courses\EE798R\End-Term Project\Code Implementation\Universal-Guided-Diffusion\stable-diffusion-guided

conda env create -f environment.yaml (No need to run again if you have already done so once)

conda activate ldm

conda install pytorch==2.2.2 torchvision==0.17.2 Pillow==10.2.0 cudatoolkit=11.3 -c pytorch -c nvidia (No need to run again once you have made the installations, true for the three statements below as well)

pip install GPUtil

pip install blobfile

pip install facenet-pytorch

# Face Recognition

mkdir test_face

"Change num_workers to 12 from 16 in the DataLoader instantiation line because of laptop limitation"

In this we generate images conditioned on the text and guided by the face identity.
The given script will take the human face from ./data/face_data/celeb/ and use it to guide the image generation conditioned on the given text prompt, set by --text

python scripts/face_detection.py --indexes 0 --text "Headshot of a person with blonde hair with space background" --optim_forward_guidance --fr_crop --optim_num_steps 2 --optim_forward_guidance_wt 200 --optim_original_conditioning --ddim_steps 50 --optim_folder ./test_face/text_type_4/ --ckpt "D:\Courses\7th sem courses\EE798R\End-Term Project\Code Implementation\Models\sd-v1-4.ckpt"

python scripts/face_detection.py --indexes 0 --text "A headshot of a woman looking like a lara croft" --optim_forward_guidance --fr_crop --optim_num_steps 2 --optim_forward_guidance_wt 200 --optim_original_conditioning --ddim_steps 50 --optim_folder ./test_face/text_type_11/ --ckpt "D:\Courses\7th sem courses\EE798R\End-Term Project\Code Implementation\Models\sd-v1-4.ckpt"

"I have tried decreasing the values of the input constants as per the inputs from TA sir, but the code still does not seem to run on my laptop. It tries opening some new kind of application and fails, thereby resulting in the text_type_4 folder forming but not populating."

The above script will produce 20 images from which one can get the top k images based on face recognition models.

python scripts/face_top_k.py --folder ./test_face/text_type_4/ --img_index 0 --img_saved 20 --top_k 5

python scripts/face_top_k.py --folder ./test_face/text_type_11/ --img_index 0 --img_saved 20 --top_k 5

# Segmentation

Images conditioned on the text and guided by segmentation maps are generated. The following script will take the images of dogs from ./data/segmentation_data/Walker/ to create the segmentation map. This segmentation map is then used to guide the image generation conditioned on the given text prompt, set by --text.

mkdir test_segmentation

python scripts/segmentation.py --indexes 1 --text "Walker hound, Walker foxhound on snow" --scale 1.5 --optim_forward_guidance --optim_num_steps 10 --optim_forward_guidance_wt 400 --optim_original_conditioning --ddim_steps 500 --optim_folder ./test_segmentation/text_type_4/ --ckpt "D:\Courses\7th sem courses\EE798R\End-Term Project\Code Implementation\Models\sd-v1-4.ckpt"

python scripts/segmentation.py --indexes 1 --text "Walker hound, Walker foxhound as an oil painting" --scale 2.0 --optim_forward_guidance --optim_num_steps 10 --optim_forward_guidance_wt 400 --optim_original_conditioning --ddim_steps 500 --optim_folder ./test_segmentation/text_type_3/ --ckpt "D:\Courses\7th sem courses\EE798R\End-Term Project\Code Implementation\Models\sd-v1-4.ckpt"

First image produced is the guiding image and the rest are generated ones.

# Object Detection

Images conditioned on text and under guidance by bounding boxes of different objects are generated. The following script will take the first sample bounding boxe in scripts/object_detection.py. The bounding boxes are then used to guide the image generation conditioned on the given text prompt, set by --text.

mkdir test_od

python scripts/object_detection.py --indexes 0 --text "a headshot of a woman with a dog" --scale 1.5 --optim_forward_guidance --optim_num_steps 5 --optim_forward_guidance_wt 100 --optim_original_conditioning --ddim_steps 250 --optim_folder ./test_od/ --ckpt "D:\Courses\7th sem courses\EE798R\End-Term Project\Code Implementation\Models\sd-v1-4.ckpt"

python scripts/object_detection.py --indexes 0 --text "a headshot of a woman with a dog on beach" --scale 1.5 --optim_forward_guidance --optim_num_steps 5 --optim_forward_guidance_wt 100 --optim_original_conditioning --ddim_steps 250 --optim_folder ./test_od/ --ckpt "D:\Courses\7th sem courses\EE798R\End-Term Project\Code Implementation\Models\sd-v1-4.ckpt"

# Style Transfer

Images generated on text conditioning and guided by style of target image is generated. The following script will take the styling images from ./data/style_folder/styles/ and use it to guide the image generation conditioned on the given text prompt, set by --text

mkdir test_segmentation

python scripts/style_transfer.py --indexes 0 --text "A colorful photo of a eiffel tower" --scale 3.0 --optim_forward_guidance --optim_num_steps 6 --optim_forward_guidance_wt 6 --optim_original_conditioning --ddim_steps 500 --optim_folder ./test_style/text_type_1/ --ckpt "D:\Courses\7th sem courses\EE798R\End-Term Project\Code Implementation\Models\sd-v1-4.ckpt"

python scripts/style_transfer.py --indexes 0 --text "A fantasy photo of volcanoes" --scale 3.0 --optim_forward_guidance --optim_num_steps 6 --optim_forward_guidance_wt 6 --optim_original_conditioning --ddim_steps 500 --optim_folder ./test_style/text_type_2/ --ckpt "D:\Courses\7th sem courses\EE798R\End-Term Project\Code Implementation\Models\sd-v1-4.ckpt"

# ImageNet Diffusion

Until now, we were using the stable diffusion model; now onwards, we use the unconditional diffusion model from OpenAI.

Therefore, we first run the following commands:

cd ..

cd D:\Courses\7th sem courses\EE798R\End-Term Project\Code Implementation\Universal-Guided-Diffusion\Guided_Diffusion_Imagenet

python Guided/Clip_guided.py --trials 5 --samples_per_diffusion 2 --text "English foxhound by Edward Hopper" --optim_forward_guidance --optim_forward_guidance_wt 2.0 --optim_num_steps 10 --optim_folder ./Clip_btd_cake/ --batch_size 8 --attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True --model_path <Path to the unconditional diffusion model>

python Guided/Clip_guided.py --trials 5 --samples_per_diffusion 2 --text "Van Gogh Style" --optim_forward_guidance --optim_forward_guidance_wt 5.0 --optim_num_steps 5 --optim_folder ./Clip_btd_cake/ --batch_size 8 --attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True --model_path <Path to the unconditional diffusion model>

python Guided/Clip_guided.py --trials 5 --samples_per_diffusion 2 --text "Birthday Cake" --optim_forward_guidance --optim_forward_guidance_wt 2.0 --optim_num_steps 10 --optim_folder ./Clip_btd_cake/ --batch_size 8 --attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True --model_path <Path to the unconditional diffusion model>

