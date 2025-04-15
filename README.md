# TV-VAD
This is the official Pytorch implementation of our paper:"Temporal-frequency and vision-language assisted for weakly supervised video anomaly detection"
![image](https://github.com/user-attachments/assets/a6f799bd-16ff-4dc1-a7ee-01a99e119fd4)

## Highlight
* We propose a Temporal GCN module and a F3R module to perform temporal modeling and enhance global frame-level features, respectively. The temporal-GCN module captures global temporal information through a transformer encoder and a lightweight GCN module. The F3R module effectively addresses the limitations in capturing global frame-level features. To the best of our knowledge, we are the first to introduce Fourier frequency-domain analysis into the WSVAD task.

* We introduce text prompts as classification anchors and use KL divergence to optimize the similarity between extracted visual and textual classification anchors features. This Video-Text alignment module with KL divergence further improves the alignment accuracy of video events description text and video frames by leveraging the anchoring role of the textual modality.

* Extensive experiments have been done to validate the proposed TV-VAD. Notably, TV-VAD accomplishes unprecedented outcomes by registering 85.64% AP on XD-Violence and 88.12% AUC on UCF-Crime respectively.
