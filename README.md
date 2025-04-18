# TV-VAD
This is the official Pytorch implementation of our paper:"Temporal-frequency and vision-language assisted for weakly supervised video anomaly detection"
![image](https://github.com/user-attachments/assets/a6f799bd-16ff-4dc1-a7ee-01a99e119fd4)

## Highlight
* We propose a Temporal GCN module and a F3R module to perform temporal modeling and enhance global frame-level features, respectively. The temporal-GCN module captures global temporal information through a transformer encoder and a lightweight GCN module. The F3R module effectively addresses the limitations in capturing global frame-level features. To the best of our knowledge, we are the first to introduce Fourier frequency-domain analysis into the WSVAD task.

* We introduce text prompts as classification anchors and use KL divergence to optimize the similarity between extracted visual and textual classification anchors features. This Video-Text alignment module with KL divergence further improves the alignment accuracy of video events description text and video frames by leveraging the anchoring role of the textual modality.

* Extensive experiments have been done to validate the proposed TV-VAD. Notably, TV-VAD accomplishes unprecedented outcomes by registering 85.64% AP on XD-Violence and 88.12% AUC on UCF-Crime respectively.

## Data preparation
* First, you need to download the datasets following:https://github.com/nwpu-zxr/VadCLIP/tree/main and put them in the direction：```./dataset```
* Then, you need to change the path of datasets. run:
  
  ```./list/rectify_csv.py```

  By the way, you need to change the path in the fourth line of the code in `./list/rectify_csv.py` to your own.

## Training
After data preparation, you can train the model on tow datasets by following command:

``` python ./main/xd_train_2.py ```

``` python ./main/ucf_train.py ```

## Testing
After training, you can evaluate the model on tow datasets by following command:

``` python ./main/xd_test.py ```

``` python ./main/ucf_test.py ```
