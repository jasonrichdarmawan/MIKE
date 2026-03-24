Connect to Lambda instance (replace <lambda-ip> with the actual IP address of your Lambda instance):
 ssh -i "C:/Users/thana/.ssh/id_rsa" ubuntu@

Install Claude Code Assistant (optional, for better terminal experience):
 curl -fsSL https://claude.ai/install.sh | bash    

Here are the commands to run sequentially on the Lambda instance:

  ---
  1. Clone the repo
  git clone --branch kim-branch --single-branch https://github.com/jasonrichdarmawan/CaKE.git ~/CaKE
  cd ~/CaKE

  ---
  2. Install dependencies
  pip install --upgrade pip

  pip install \
      torch==2.4.0 \
      torchvision==0.19.0 \
      --index-url https://download.pytorch.org/whl/cu121

  pip install \
      transformers==4.44.2 \
      accelerate==0.34.2 \
      sentencepiece==0.2.0 \
      pandas==2.2.2 \
      numpy==1.26.4 \
      matplotlib==3.9.2 \
      seaborn==0.13.2

  Verify GPU and versions:
  python3 -c "
  import torch, transformers, pandas, numpy
  print('torch         ', torch.__version__, '  CUDA:', torch.version.cuda)
  print('transformers  ', transformers.__version__)
  print('pandas        ', pandas.__version__)
  print('numpy         ', numpy.__version__)
  print('GPU available:', torch.cuda.is_available())
  if torch.cuda.is_available():
      p = torch.cuda.get_device_properties(0)
      print('GPU:          ', p.name, f'  {p.total_memory/1e9:.1f} GB')
  "

  ---
  3. HuggingFace login (Llama only, skip for GPT-2)
  python3 -c "from huggingface_hub import login; login('YOUR_HF_TOKEN_HERE')"

  ---
  4. Run the attribution (GPT-2)
  cd ~/CaKE/analysis

  python3 neuron_attribution.py \
      --model-name gpt2 \
      --data-dir ~/CaKE/datasets/gpt2 \
      --subsets correct incorrect inconsistent \
      --prompts two-hop first-hop second-hop \
      2>&1 | tee ~/neuron_attribution_gpt2.log

  Or for Llama-3-8B:
  cd ~/CaKE/analysis

  python3 neuron_attribution.py \
      --model-name meta-llama/Meta-Llama-3-8B \
      --data-dir ~/CaKE/datasets/Meta-Llama-3-8B \
      --subsets correct incorrect inconsistent \
      --prompts two-hop first-hop second-hop \
      2>&1 | tee ~/neuron_attribution_llama3.log

  ---
  5. Copy results back to your local machine (run from your local terminal)
  rsync -avz --progress --exclude '*_ckpt.pkl' \
      ubuntu@<lambda-ip>:~/CaKE/datasets/gpt2/neuron_attribution/ \
      ./datasets/gpt2/neuron_attribution/