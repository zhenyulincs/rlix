cd external/ROLL_schedrl
conda activate main # ensure we are in main env 
uv pip install -r requirements_torch260_vllm.txt
uv pip install transformer-engine[pytorch]==2.2.0
curl -fsSL https://opencode.ai/install | bash
curl -fsSL https://claude.ai/install.sh | bash
npm i -g @openai/codex
bash -c "$(curl -fsSL https://gitee.com/iflow-ai/iflow-cli/raw/main/install.sh)"
