## Speech BabyLM ABX 

This code uses [fastabx](https://github.com/bootphon/fastabx
) ([arxiv](https://arxiv.org/pdf/2505.02692)) and representation extraction code from [internal_tools](https://github.com/mdhk/internal_tools.git) to calculate and plot ABX for a variety of models. Currently we evaluate on the ZeroSpeech2021 triphone ABX and on the recently released [prosodic ABX](https://arxiv.org/pdf/2505.02692). For ZeroSpeech we evaluate on a sample of speakers, as the dataset is pretty big.


The project can be installed with `setup_project.py` which will, in order, do:
1. Download internal_tools at external/internal_tools/
2. Install the Python environment with uv (uv sync)
3. Download model weights at models/

Then you can prepare the ABX task files by running `src/prepare_tasks.py`

Finally, run the evaluation with either `run_all_fast.sh` or `run_all.sh`

The code is still in development, with the goal of reducing the reliance on `internal_tools`. 