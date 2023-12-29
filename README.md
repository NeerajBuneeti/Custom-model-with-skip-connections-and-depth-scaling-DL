# Group14_CS584(Machine learning)
This repository is about the project which is part of CS584 (Machine Learning). The objective of this project is to construct a custom model with lesser parameters and higher performance with skip connections and depth scaling principals. We have considered 2 SOTA models(ResNet50 and VGG19) and one custom model in this project. 

Dataset 1: Butterfly and moth species (https://www.kaggle.com/datasets/gpiosenka/butterfly-images40-species)  
Dataset 2: FER (https://www.kaggle.com/datasets/msambare/fer2013)  
ML14a: Model training on Butterfly dataset  
ML14b: Model training on FER dataset

# Problem 
The realm of very deep neural networks is marked by its immense potential for intricate tasks in computer vision. However, two significant challenges cast shadows on their widespread adoption.

# a. Vanishing Gradient:
As the depth of a neural network increases, the backpropagation of errors during training encounters a critical hurdle known as the vanishing gradient problem [10, 11, 12]. This challenge arises due to the diminishing magnitude of gradients as they traverse through numerous layers during backpropagation. In essence, the gradients dwindle to near-zero values, rendering them ineffective in updating the weights of early layers. Consequently, these layers fail to learn meaningful representations, impeding the overall convergence of the network. The vanishing gradient problem becomes particularly pronounced in very deep architectures, where the prolonged path that gradients traverse exacerbates the attenuation effect.

# b. Long Computational Time:
Simultaneously, the computational demands associated with training very deep neural networks escalate dramatically. The intricate interplay of an increasing number of layers and parameters necessitates extensive computational resources and time. The sheer complexity of the network structure contributes to prolonged training times, making it a formidable challenge to efficiently harness the potential of these deep architectures within reasonable timeframes. The extended training periods not only strain computational resources but also hinder the swift development and deployment of models for practical applications.

To address these challenges, we have introduced a custom model with fewer parameters and enhanced performance. The details of the proposed algorithm are discussed in Section 3 of our project report.  Subsequently, Model training and experimentation, Results, and discussions are presented in Sections 4 and 5, respectively. 
