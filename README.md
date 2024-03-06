# FootstrikeDetection
Code for paper ""<br/>

## Prerequisites
Install necessary libraries with `pip install -r requirements.txt`<br/>

## Dataset
Sample data is included in the `data` folder. <br/>
`rat_2_DeepEthogram_result.h5`: The result from [DeepEthogram](https://github.com/jbohnslav/deepethogram).<br/>
The DeepEthogram labels are "touch" which indicates the foot-strike onset of the forelimb, and "walk" which indicates the walking/still state of the animal.<br/>
<br/>
`rat_2_DeepLabCut_result.h5`: The result from [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut).<br/>
Three markers "f_foot",	"f_elbow",	"f_sholder" are labeled on the video frames.<br/>
<br/>
`rat_2_raw_lfp.npy`: The 32 channel LFP.<br/>
<br/>
`rat_2_trigger.npy`: The camera strobe index corresponding to the samples in `rat_2_raw_lfp.npy`.<br/>
<br/>
`rat_2_manual_labels.csv`: The onsets by manual labels.<br/>

## Running the script
The path of DeepLabCut result and DeepEthogram result along with other parameters should be specified in `config.yaml`.<br/>
To run, simply execute `python -m main.py` in the terminal or command prompt. It will output an image with an aligned LFP.<br/>
To use the script in a Python pipeline, use the function `get_valid_step_idx` in `utils.py`.<br/>

## Sample image
<p align="left">
<img src="https://github.com/UT-yakusaku/FootstrikeDetection/blob/main/aligned_lfp.png">
</p>
