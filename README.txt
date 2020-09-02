# Meta_Hyperband_implementation
This is the source code for "Meta-Hyperband: Hyperparameter optimization with meta-learning and Coarse-to-Fine" paper, accepted by IDEAL2020.
Authors: Samin Payrosangari1, Afshin Sadeghi1,2B,Damien Graux3, and Jens Lehmann1,21Department of Computer Science, University of Bonn, Germany2Fraunhofer IAIS, Sankt Augustin, Germany3ADAPT SFI Research Centre, Trinity College Dublin, Irelandsaminpayro@gmail.com, sadeghi@cs.uni-bonn.de,damien.graux@adaptcentre.ie, jens.lehmann@cs.uni-bonn.de


Abstract. Hyperparameter optimization is one of the main pillars of ma-chine learning algorithms. In this paper, we introduce Meta-Hyperband: a Hyperband based algorithm that improves the hyperparameter optimization by adding levels of exploitation. Unlike Hyperband method, which is a pure exploration bandit-based approach for hyperparameter optimization, our meta approach generates a trade-off between exploration and exploitation by combining the Hyperband method with meta-learning and Coarse-to-Fine modules. We analyze the performance of Meta-Hyperband on various datasets to tune the hyperparameters of CNN and SVM. The experiments indicate that in many cases Meta-Hyperband can discover hyperparameter configurations with higher quality than Hyperband, using similar amounts of resources. In particular, we discovered a CNN configuration for classifying CIFAR10 dataset which has a 3% higher performance than the configuration founded by Hyperband and is also 0.3% more accurate than the best-reported configuration of the Bayesian optimization approach. Additionally, we release a publicly available pool of historically well-performed configurations on several datasets for CNN and SVM to ease the adoption of Meta-Hyperband.

How to: 
To use this implementation of Meta-Hyperband you need to configure some functions and variables in the hyper.py as well as meta-hyperband.py scripts which are explained as comments in the scripts as well.
#------------------------
try_params(step_i,n_resources,params,save_path) : 
The hyper.py script defines a function called: try_params(step_i,n_resources,params,save_path)
This function is where your ML algorithm is called for training on the dataset you define as well as the amount of resources you want to spend for this training. step_i identifies in which bracket of Meta-Hyperband are you doing this evaluation. In the current script, the 18% layer of cuda-convnet CNN is sampled which was also used in our experiments. For cuda-convnet CNN datasets should be prepared into batches as mentioned in the cuda-convnet documentation as well. Other ML models can be called here according to the desired experiments.
#------------------------
get_space(params={}) :
At the begining of hyper.py the get_space(params={}) function is defined. here you can define the initial space for sampling hyperparameters out of it. in case you already pass a hyperparameter configuration via params={} variable to it, this config will be treated as a Coarse and will be fine-tuned via Coarse-to-Fine module and returns a restricted space--> this is how this function is called from Bracket 3 and Bracket 0 of meta-hyperband.py script. 
#------------------------
get_params(space,s,j):

This function samples hyperparameter configurations from the space randomly--> this space can be main space during random search brackets or Coarse-to-Fine restricted space which is passed.
#------------------------
get_meta_params(section,s,j,dataset_name,save_path):
this function gets the historical configurations from meta-learning pool (in this project pool-for-CNN.cfg or pool-for-SVM.cfg). The meta-learning pool should be in cfg format. If this pool doesn't exist for your ML algorithm, the bracket 1 part in the meta-hyperband.py should be modified to do random search.
#------------------------
save_results(result, name_dataset,num_dataset):
This function saves the best discovered configuration discovered in the current run of Meta-Hyperband hyperparameter optimization in the pool of meta-learning.
#------------------------
The meta-hyperband.py script defined the Meta-Hyperband algorithm, including it's brackets, and calls the functions discussed above.
#------------------------
Note: The 18% layer is taken from the cuda-convnet directory however, the implementation of the cuda-convent2 has been applied to this layer (as cuda-convnet2 was the newer version at the time of this project) .Moreover, In the cuda-convnet2 original code you can define how many epochs you want to run the training, however in this paper we didn't want to train all the configurations for the whole epochs but using adaptive resource allocations we wanted to train them starting from several iterations (that’s the inherit of Meta-Hyperband and its parent algorithm Hyperband), so we had to modify the cuda-convnet code to be able to get the number of training iterations as input.
