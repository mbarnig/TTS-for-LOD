# TTS for LOD
## A synthetic voice for the luxembourgish online dictionary
#### References : my earlier public Text-to-Speech projects hosted at Github and HuggingFace  
* [Multilingual-TTS](https://github.com/mbarnig/Multilingual-TTS)
* [Marylux-648-TTS-Corpus](https://github.com/mbarnig/Marylux-648-TTS-Corpus)
* [TTS Model lb-de-fr-en-pt-coqui-vits-tts](https://huggingface.co/mbarnig/lb-de-fr-en-pt-coqui-vits-tts)
* [Mir schwätzen och Lëtzebuergesch !](https://huggingface.co/spaces/mbarnig/lb_de_fr_en_pt_COQUI_VITS_TTS)
#### New high-quality dataset
To train a high-quality luxembourgish TTS voice, the [ZLS](https://portal.education.lu/zls) (Zenter fir d'Lëtzebuerger Sprooch) assembled an outstanding luxembourgish dataset of 39.836 audio samples, with related transcriptions, recorded in studio quality by Max Kuborn. After the first training I discovered that the dataset included several hundred files with a female voice. I cleaned the dataset and retrained the model with 32.000 male samples. My Wiki-Page [Dataset](https://github.com/mbarnig/TTS_for_LOD/wiki/Dataset) provides detailed informations about this corpus.
#### Processing environment
The main hardware required for the TTS-training is a NVIDIA graphic card. My ancient TTS development system was set-up two years ago in a Linux-Ubuntu desktop with a NVIDIA RTX2070 card. When I started the training with the new dataset I was disappointed because my old scripts were no longer working without errors, probably due to automatic updates of some Python modules which are no longer compliant with my original configuration. After some frustrating attempts to handle the errors, I decided to restart from scatch and to set-up a new developement system on a Windows 11 labtop with a NVIDIA card RTX3060. My Wiki-Page [Processing Environment](https://github.com/mbarnig/TTS-for-LOD/wiki/Processing-Environment) shows how to install all the required sofware.
#### VITS TTS model
The choice of the Coqui-TTS-VITS model is explained in my Wiki-Page [TTS-Model](https://github.com/mbarnig/TTS-for-LOD/wiki/TTS-Model).
#### Installation of the Coqui-TTS Tools
Creating the required developement environment on a personal computer takes some time and can be frustrating. I published a small guide on the Wiki-Page [Coqui-TTS Tools](https://github.com/mbarnig/TTS-for-LOD/wiki/Coqui%E2%80%90TTS-Tools).
#### Training script
The training script is the heart of the project. All details are provided at [Training Script](https://github.com/mbarnig/TTS-for-LOD/wiki/Training-Script).
#### Training process
It's important to understand the [Training Process](https://github.com/mbarnig/TTS-for-LOD/wiki/Training-Process).
#### Evaluation
The continuous evaluation of the training evolution is the hard job. I will extend the [Evaluation Guide](https://github.com/mbarnig/TTS-for-LOD/wiki/Evaluation) as soon as possible.
#### Tensorboard
I will add more figures to my Wiki-Page [Tensorboard](https://github.com/mbarnig/TTS-for-LOD/wiki/TensorBoard).
#### Best Model
These details will be provided at the end of the training in the Wiki-Page [Best Model](https://github.com/mbarnig/TTS-for-LOD/wiki/Best-Model).
#### Inference
When the model is ready, it can be used to synthesize luxembourgish texts. The scripts to run the synthesis and the access to public demo-spaces will be explained in the [Inference Guide](https://github.com/mbarnig/TTS-for-LOD/wiki/Inference).
