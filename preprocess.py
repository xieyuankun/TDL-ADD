import raw_dataset
from feature_extraction import *
import os
import torch
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2Tokenizer
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
device = torch.device("cuda" if cuda else "cpu")

def pad_dataset(wav):
    waveform = wav.squeeze(0)
    waveform_len = waveform.shape[0]
    cut = 64600
    if waveform_len >= cut:
        waveform = waveform[:cut]
        return waveform
    # need to pad
    num_repeats = int(cut / waveform_len) + 1
    padded_waveform = torch.tile(waveform, (1, num_repeats))[:, :cut][0]
    return padded_waveform

def normalization(orign_data):
    d_min = orign_data.min()
    if d_min < 0:
        orign_data += torch.abs(d_min)
        d_min = orign_data.min()
    d_max = orign_data.max()
    distance = d_max - d_min
    norm_data = (orign_data - d_min).true_divide(distance)
    return norm_data

for part_ in ["train", "dev", "eval"]:
    asvspoof_raw = raw_dataset.ASVspoof2019PSRaw("/home/xieyuankun/data/asv2019PS",
                                           "/home/xieyuankun/data/asv2019PS/ASVspoof2019_PS_cm_protocols/", part=part_)
    target_dir = os.path.join("/home/xieyuankun/data/asv2019PS/preprocess_xls-r-300m", part_,
                              "xls-r-300m")
    # mel = wav2vec2_large_CTC()
    processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-300m")  # best
    # Wav2Vec2FeatureExtractor =Wav2Vec2FeatureExtractor(feature_size=1024)
    # feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1024).from_pretrained("facebook/wav2vec2-base-960h")
    # model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h").extrcatfeatures.cuda()
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m").cuda()
    # model.eval()
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for idx in tqdm(range(len(asvspoof_raw))):
        waveform, filename = asvspoof_raw[idx]
        waveform = waveform.to(device)
        print(waveform.shape, 'waveform')
        waveform = waveform.squeeze(dim=0)
        # waveform = pad_dataset(waveform).to('cpu')
        input_values = processor(waveform, sampling_rate=16000,
                                 return_tensors="pt").input_values.cuda()
        with torch.no_grad():
            wav2vec2 = model(input_values).last_hidden_state.cuda()
        print(wav2vec2.shape)
        torch.save(wav2vec2, os.path.join(target_dir, "%s.pt" % (filename)))
    print("Done!")