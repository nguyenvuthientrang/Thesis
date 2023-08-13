##  Attack on Backdoor: Backdoor Attacks in Prompt-based Continual Learning
PyTorch code for the Thesis:\
**Attack on Backdoor: Backdoor Attacks in Prompt-based Continual Learning**\
**_Nguyễn Vũ Thiên Trang_**, DSAI64, School of Information and Communication Technology, Hanoi University of Science and Technology.

## Abstract
Continual Learning has emerged as a promising solution for meeting data privacy and security requirements, as it enables continuous learning from diverse data sources without storing data. However, the fundamental mechanism of continual learning, designed to overcome catastrophic forgetting, inadvertently creates a potential vulnerability that can be exploited by sophisticated attack methods like backdoor attacks. A backdoor attack exploits model vulnerabilities to inject malicious knowledge, causing the model to misbehave when exposed to specific triggers in the data. In this groundbreaking thesis, the potentiality of targeted backdoor attacks within the context of continual learning is delved into, and the persistence and adaptability of backdoor knowledge in an incremental learning process are investigated. Specifically, the thesis proposes a novel backdoor attack framework, named \emph{Attack-on Backdoor (AOB)}, tailored for prompt-based continual learning, which guides poisoned data to select malicious prompts. The framework leverages bilevel optimization to simulate a continual learning process, optimizing triggers to align with malicious prompts. Furthermore, an orthogonality loss is applied to enhance the distinctiveness between backdoor prompts and clean prompts, and binary cross-entropy is employed to establish an independent trigger optimization mechanism. Extensive experimentation confirms the significant threat of backdoor attacks to continual learning, with an astonishing attack success rate reaching 100\%. The source code for the thesis is publicly available at \url{https://github.com/nguyenvuthientrang/Thesis}.

## Setup
 * Install anaconda: https://www.anaconda.com/distribution/
 * set up conda environmet w/ python 3.8, ex: `conda create --name coda python=3.8`
 * `conda activate coda`
 * `sh install_requirements.sh`
 * <b>NOTE: this framework was tested using `torch == 2.0.0` but should work for previous versions</b>


## Training
To execute each phase of the attacking pipeline, make the following modifications:

1. Modify the `run.py` file.
2. Adjust the configuration files in the `configs/` directory.
3. Update the script files in the `experiments/` directory.


```bash
sh experiments/cifar100.sh
```

## Results
Results will be saved in a folder named `outputs/`. To get the final average accuracy, retrieve the final number in the file `outputs/**/results-asr/pt.yaml`


## Acknowledgement
This implementation is based on [ruoxi-jia-group/Narcissus](https://github.com/ruoxi-jia-group/Narcissus) and [GT-RIPL/CODA-Prompt](https://github.com/GT-RIPL/CODA-Prompt).
