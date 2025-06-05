# 4K Video Generation Pipeline (Wan2.1-I2V)

This repository provides a custom tile-scale pipeline for 4K video generation, building upon [Wan2.1-I2V](https://github.com/Wan-Video/Wan2.1).

## Instructions

### 1. Clone the Official Repository

```bash
git clone https://github.com/Wan-Video/Wan2.1
cd Wan2.1
```

### 2. Set Up the Environment

Follow the official instructions in the Wan2.1 repository to install the required dependencies. Make sure you use the I2V-14B-720P or I2V-14B-480P model when generating videos.

### 3. Move the Provided Pipeline

Move the provided `image2video_tilescale.py` into the pipelines directory, replacing the existing file:

```bash
cd Wan2.1
cp ./image2video_tilescale.py ./wan/image2video.py
```

### 4. Move the Provided Model, Attention

Move the provided `model_tilescale.py` and `attention_tilescale.py` into the modules directory, replacing the existing file:

```bash
cd Wan2.1
cp ./model_tilescale.py ./wan/modules/model.py
cp ./attention_tilescale.py ./wan/modules/attention.py
```

### 5. Run the Video Generation

Use the standard scripts for video generation as described in the official repository.  
Ensure you specify the appropriate parameters for 4K output.

```bash
cd Wan2.1
# Here's the default command line as in HunyuanVideo-I2V's official repository.
python python generate.py 
    --task i2v-14B \
    --size 1280*720 \
    --ckpt_dir ./Wan2.1-I2V-14B-720P \
    --image examples/i2v_input.JPG \
    --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. \The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```