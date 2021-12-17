<div align="center">    

# STEPS: <u>S</u>emantic <u>T</u>yping of <u>E</u>vent <u>P</u>rocesses with a Sequence-to-Sequence Approach

[![Paper](https://img.shields.io/badge/proc-AAAI--Proceedings-blue)](https://github.com/SapienzaNLP/steps/blob/main/docs/AAAI22_STEPS.pdf)
[![Conference](https://img.shields.io/badge/aaai-AAAI--2022-red)](https://aaai.org/Conferences/AAAI-22/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

</div>


If you find our paper, code or framework useful, please reference this work in your paper:

```
@inproceedings{pepe-etal-2021-steps,
  title     = {STEPS: Semantic Typing of Event Processes with a Sequence-to-Sequence Approach},
  author    = {Pepe, Sveva and Barba, Edoardo, and Blloshmi, Rexhina and Navigli, Roberto},
  booktitle = {Proceedings of {AAAI}},
  year      = {2022},
}
```

<!-- ## Code Release  -->

<!-- * The code for running the experiments of STEPS will be released soon! -->

## Environment Setup

Requirements:

* Debian-based (e.g. Debian, Ubuntu, ...) system
* [conda](https://docs.conda.io/en/latest/)

We strongly advise utilizing the bash script setup.sh to set up the python environment for this project.
Run the following command to quickly setup the env needed to run our code: 

```
bash ./setup.sh
```
It's a bash command that will setup a conda environment with everything you need. Just answer the prompts as you proceed.

## Checkpoint

* Checkpoint of STEPS will be released soon!

## Train

Training is done via the training script, src/train.py, and its parameters are read from the .yaml files in the conf/ folders. Once you applied all your desired changes, you can run the new training with:

```
(steps) user@user-pc:~/steps$ PYTHONPATH=$(pwd) python src/train.py
```

## Evaluate

If you want to evaluate the model you just have to run the following command:

```
(steps) user@user-pc:~/steps$ PYTHONPATH=$(pwd) python src/predict.py --ckpt <steps_checkpoint.ckpt>
```

## License
This project is released under the CC-BY-NC-SA 4.0 license (see `LICENSE`). If you use `STEPS`, please put a link to this repo.

## Acknowledgements
The authors gratefully acknowledge the support of the [ERC Consolidator Grant MOUSSE](http://mousse-project.org) No. 726487 and the [ELEXIS project](https://elex.is/) No. 731015 under the European Union’s Horizon 2020 research and innovation programme.

* This work was supported in part by the MIUR under the grant "Dipartimenti di eccellenza 2018-2022" of the Department of Computer Science of the Sapienza University of Rome.
