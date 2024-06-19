# TTS for LOD
## A synthetic voice for the luxembourgish online dictionary
#### References : my earlier public Text-to-Speech projects hosted at Github and HuggingFace  
* [Multilingual-TTS](https://github.com/mbarnig/Multilingual-TTS)
* [Marylux-648-TTS-Corpus](https://github.com/mbarnig/Marylux-648-TTS-Corpus)
* [TTS Model lb-de-fr-en-pt-coqui-vits-tts](https://huggingface.co/mbarnig/lb-de-fr-en-pt-coqui-vits-tts)
* [Mir schwätzen och Lëtzebuergesch !](https://huggingface.co/spaces/mbarnig/lb_de_fr_en_pt_COQUI_VITS_TTS)
#### New high-quality dataset
To train a high-quality luxembourgish TTS voice, the [ZLS](https://portal.education.lu/zls) (Zenter fir d'Lëtzebuerger Sprooch) assembled an outstanding luxembourgish dataset of 39.836 audio samples, with related transcriptions, recorded in studio quality by Max Kuborn. My Wiki-Page [Dataset](https://github.com/mbarnig/TTS_for_LOD/wiki/Dataset) provides detailed informations about this corpus.
#### Processing environment
The main hardware required for the TTS-training is a NVIDIA graphic card. My ancient TTS development system was set-up two years ago in a Linux-Ubuntu desktop with a NVIDIA card. When I started the training with the new dataset I was disappointed because my old scripts were no longer working without errors, probably due to automatic updates of some Python modules which are no longer compliant with my original configuration. After some frustrating attempts to handle the errors, I decided to restart from scatch and to set-up a new developement system on a Windows 11 labtop with a NVIDIA card . My Wiki-Page [Processing Environment](https://github.com/mbarnig/TTS-for-LOD/wiki/Processing-Environment) shows how to install all the required sofware.

Installation of the Coqui-TTS Tools

Training scripts

Training process

Evaluation

Tensorboard

Best Model

Inference

