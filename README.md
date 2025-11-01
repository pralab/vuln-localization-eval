[![Generic badge](https://img.shields.io/badge/Springer-Machine_Learning-orange)]()
[![arXiv](https://img.shields.io/badge/arXiv-2510.11202-b31b1b.svg)](https://arxiv.org/abs/2510.11202)
<div align="center">
<h1>Evaluating Line-level Localization Ability of Learning-based Code Vulnerability Detection Models</h1>

We introduce <em>Detection Alignment (DA)</em>, a metric that quantifies how well ML models localize vulnerabilities to specific vulnerable lines of code.

<img src="/imgs/graphical_abstract.svg" alt="Graphical Abstract" width="80%" height="auto">

The implementation of the DA metric can be found in the [file](https://github.com/pralab/vuln-localization-eval/detection_alignment.py). For a complete explanation, please refer to the paper.
</div>
<div align="left">

<h1>How To Use</h1>

- [Set up Environment](#set-up-environment)
- [Download dataset and models](#download-dataset-and-pretrained-models)
- [Evaluate your vulnerability detector with DA](#run-da-evaluation)
- [Reproduce the experiments](#reproduce-experiments)
- [Cite our work](#bibtex-citation)

</div>

## Set up Environment 
```bash
# Clone the repository
git clone https://github.com/pralab/vuln-localization-eval.git
cd vuln-localization-eval

# (Optional) Create a conda environment
conda create -n da-evaluation python=3.10 -y
conda activate da-evaluation

# Install dependencies
pip install -r requirements.txt
```

## Download dataset and pretrained models
Work in progress

## Run DA Evaluation
Work in progress

## Reproduce experiments
Example of CodeBERT usage that calculates line-level scores using attention values from encoder layer 0.
```bash
python main_exp.py \
  --test_data_file '/your/path/to/BigVul' \
  --model_path '/your/path/to/pretrained/model' \
  --model_type 'codexglue' \
  --block_size 512 \
  --block_index 0 \
  --device 'cpu' \
  --xai_method 'attention' \
  --seed 42 \
  --vuln_threshold 0.5 \
```

## Contact
We welcome questions, suggestions, and contributions. Please open an issue or pull request to get in touch.

## BibTex citation
```bibtex
@article{pintore2025evaluating,
  title={Evaluating Line-level Localization Ability of Learning-based Code Vulnerability Detection Models},
  author={Pintore, Marco and Piras, Giorgio and Sotgiu, Angelo and Pintor, Maura and Biggio, Battista},
  journal={arXiv preprint arXiv:2510.11202},
  year={2025}
}
```
## Acknowledgements
This work has been partly supported by the EU-funded Horizon Europe projects [ELSA – European Lighthouse on Secure and Safe AI](https://elsa-ai.eu) (GA no. 101070617)
and [Sec4AI4Sec - Cybersecurity for AI-Augmented Systems](https://www.sec4ai4sec-project.eu) (GA no. 101120393); and by projects [SERICS](https://serics.eu/) (PE00000014) and FAIR (PE00000013, CUP:
J23C24000090007) under the MUR NRRP funded by the European Union - NextGenerationEU.

<img src="imgs/SERICS.png" alt="serics" style="width:200px;"/> &nbsp;&nbsp; 
<img src="imgs/sec4AI4sec.png" alt="sec4ai4sec" style="width:70px;"/> &nbsp;&nbsp; 
<img src="imgs/elsa.jpg" alt="elsa" style="width:70px;"/> &nbsp;&nbsp;
<img src="imgs/SAFER_Logo.png" alt="elsa" style="width:70px;"/> &nbsp;&nbsp;
<img src="imgs/FundedbytheEU.png" alt="LInf" style="width:240px;"/>
