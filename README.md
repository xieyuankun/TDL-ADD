# TDL for partially deepfake detection
This is the pytorch implementation of our work titled "An Efficient Temporary Deepfake Location Approach Based Embeddings for Partially Spoofed Audio Detection ," which was available on arxiv at "https://arxiv.org/abs/2309.03036".

##  1. Offline Data Extraction
Please download the training, development and evaluation set from [ASVspoof PartialSpoof Database](https://zenodo.org/records/5766198) first.

Please ensure the data and label position are correct. If you need to adjust, please modify the calss ASVspoof2019PSRaw in raw_dataset.py.

After downloading, plase place the train, dev, and eval raw wave folders of 19PS under `/home/xieyuankun/data/asv2019PS/`.

We have provided the padded label in `./label/`. Please place the train, dev, and eval labels under `/home/xieyuankun/data/asv2019PS/ASVspoof2019_PS_cm_protocols/`.

After preprocess, the last hidden states of wav2vec2 will be saved in `/home/xieyuankun/data/asv2019PS/preprocess_xls-r-300m`.


## 2. Train Model

```
python main_train.py 
```
Before running the `main_train.py`, please change the `path_to_features` according to the files' location on your machine.

If training is slow, consider adjusting the num_worker parameter in conjunction with the number of CPU cores. 
The default is set to 8. If performance remains slow, you may explore multi-GPU training in args.

## 3. Test
```
python generate_score_offline.py 
python eval_ps.py
```
You will get the final test EER, Precision, Recall and F1_score.
