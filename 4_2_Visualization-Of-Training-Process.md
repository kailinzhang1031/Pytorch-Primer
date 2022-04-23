# 4.2 Visualization Of Training Process

# 1. Code

Please check in the previous session.

# 2. Illustration

# 2.1 Methods In TensorboardX

| Function | Description | Call |
| :---: | :---: | :---: |
| SummaryWriter | | writer = SummaryWriter() |
| writer.add_scalar() | | writer.add_scalar('myscalar', value, iteration) | 
| writer.add_image() | | writer.add_image('imresult',x,iteration) |
| writer.add_histogram() | | writer.add_histogram('hist',array, iteration) |
| writer.add_graph() | | writer.add_graph(model, input_to_model = None) |
| writer.add_audio() | | writer.add_audio(tag, audio, iteration, sampler_rate) |
| writer.add_text() | | writer.add_text(tag, text_string, global_step=None) |


## 2.2 Explore Visualization

We can launch terminal and run the command: tensorboard --logdir='data/log'.

Figure 1 displays loss and accuracy in the training process.

[Figure1](Images/4_3_1_Visualization-of-training-process-using-tensorboardX-1.jpg)

Figure 2 displays images in a batch.

[Figure2](Images/4_3_2_Visualization-of-training-process-using-tensorboardX-2.jpg)

Figure 3 displays distributions of parameters.

[Figure3](Images/4_3_3_Visualization-of-training-process-using-tensorboardX-3.jpg)

Figure 4 displays histograms of parameters.


[Figure4](Images/4_3_4_Visualization-of-training-process-using-tensorboardX-4.jpg)

Figure 5 displays time series in the training process.

[Figure5](Images/4_3_5_Visualization-of-training-process-using-tensorboardX-5.jpg)
