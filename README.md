# Improving the Effective Receptive Field of Message-Passing Neural Networks [ICML 2025]

[Shahaf E. Finder](https://shahaffind.github.io/), [Ron Shapira Weber](https://ronshapiraweber.github.io/), [Moshe Eliasof](https://science.ai.cam.ac.uk/team/moshe-eliasof), [Oren Freifeld](https://www.cs.bgu.ac.il/~orenfr/) and [Eran Treister](https://www.cs.bgu.ac.il/~erant/)

[![arXiv](https://img.shields.io/badge/arXiv-2505.23185-b31b1b.svg?style=flat)](https://arxiv.org/abs/2505.23185)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/improving-the-effective-receptive-field-of/node-classification-on-pascalvoc-sp-1)](https://paperswithcode.com/sota/node-classification-on-pascalvoc-sp-1?p=improving-the-effective-receptive-field-of)  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/improving-the-effective-receptive-field-of/node-classification-on-coco-sp)](https://paperswithcode.com/sota/node-classification-on-coco-sp?p=improving-the-effective-receptive-field-of)

## Updates
* June 11, 2025 - Added City-networks training script, and GatedGCNConv option.

## How to use
Example:
```python
from models.immpnn import IMMPNN

model = IMMPNN(
  in_channels=37,
  hidden_channels=32,
  out_channels=10,
  num_layers=16,
  scales=3,
  dropout=0.2,
  conv_type='gcnconv',
)
```

Code for additional message-passing protocols, as well as ready-to-use code for all the experiments from the paper, will be released in the near future.

### City-networks
For information about the dataset see the [original repo](https://github.com/LeonResearch/City-Networks).

The training script is provided in `main_city_networks.py`. The command used to run the experiments:
```bash
model_layers=16
scales=4
dataset_name=paris

for seed in {0..4}; do
    python main.py --dataset $dataset_name --seed $seed --model imgcn --model-layers $model_layers --scales $scales --wandb --run-name $dataset_name-imgcn-$model_layers-layers-$scales-scales-$seed-seed
done
```

## License
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.

## Citation
If you find this repository helpful, please consider citing:
```
@inproceedings{finder2025improving,
  title     = {Improving the Effective Receptive Field of Message-Passing Neural Networks},
  author    = {Finder, Shahaf E and Shapira Weber, Ron and Eliasof, Moshe and Freifeld, Oren and Treister, Eran},
  booktitle = {International Conference on Machine Learning},
  year      = {2025},
}
```
