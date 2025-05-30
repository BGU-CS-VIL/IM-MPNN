# Improving the Effective Receptive Field of Message-Passing Neural Networks [ICML 2025]

[Shahaf E. Finder](https://shahaffind.github.io/), [Ron Shapira Weber](https://ronshapiraweber.github.io/), [Moshe Eliasof](https://science.ai.cam.ac.uk/team/moshe-eliasof), [Oren Freifeld](https://www.cs.bgu.ac.il/~orenfr/) and [Eran Treister](https://www.cs.bgu.ac.il/~erant/)

### How to use
Import the model, for example:
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
