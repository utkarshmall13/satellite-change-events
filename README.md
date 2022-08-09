# Code for [Change Event Dataset for Discovery from Spatio-temporal Remote Sensing Imagery](https://www.cs.cornell.edu/projects/satellite-change-events/)
---

The code is divided 3 parts
1. Obtaining Sentinel-2 Data
2. Using Self-supervised Pipeline
3. Baselines

The code was tested on python 3.8. use `requirements.txt.` to install python.
> pip install -r /path/to/requirements.txt.
---
## Obtaining Sentinel-2 Data

Change the directory to `data`.
To download the Raw input for CaiRoad use `sentinel_cairo.py`, for CalFire use `sentinel_cali.py`.
> python3 sentinel_cairo.py

This code requires EarthEngine api, where an account needs to be created for authentication. Follow the from [here](https://developers.google.com/earth-engine/guides/python_install) for installation and authentication.

This also downloads the cloud masks for the dataset.

Note that the raw data obtained from this is already there on the [project page](https://www.cs.cornell.edu/projects/satellite-change-events/). 
The directory is named `fulldata`in the downloaded `CaiRoad.zip` or ``CalFire.zip``
---
## Using Self-supervised Pipeline
Change the directory to `pipeline`.
The pipeline itself has 3 parts:

### Training and running pairwise change detection
To train our self-supervised change detection, use `train_SSNet.py`.
> python3 train_SSNet.py -bf -fr -m train

The default batch size is set to 4. But on larger GPUs batch size can be increased.
Instead of training, trained models can be downloaded from the [project page](https://www.cs.cornell.edu/projects/satellite-change-events/static/models/).

For inferring use the same switches in the argument.
> python3 train_SSNet.py -bf -fr -m inference

This will create a directory with binary change detection. The next step is grouping the changes.

### Change grouping.

Use `region_growing.py` to group the changes.
> python3 region_growing.py

This will create a directory with grouped change events. The changes are stored in two formats 1) a `.npy` file for storing the data and 2) `png` files fo visualization.

The change events are not yet in the  standard format. Follow the next step for that.

### Obtaining change events.

Use `get_change_events.py` to get individual change events from the change grouping.
> python3 get_change_events.py

Alternatively, this data can also be downloaded from the project page.

**The above mentioned pipeline can be used for any set of region to obtain summarizing change events from them.**

---
## Baselines

Change the directory to `baselines`.
Use `main.py` to run different baselines. For example use the following for SimCLR:Change Events :
> python3 main.py -dn CalFire -mt all -bs 256 -e 20

Change `-dn` to `CaiRoad` to run on CaiRoad dataset.
Use arguments our of `notime`, `nochange`, `fulldata`, `eurosat`, `imnet` for `-mt` to run other baselines.
The trained models are stored in a directory named `models`, that can used for inference. 
For inference use argument `-m infer`.

The reported results can be viewed using this.