#The code can be found at: https://github.com/saminpayro/Meta_Hyperband_implementation
#Also used another code as reference: Copyright (c) 2017, Zygmunt Zając, https://github.com/zygmuntz/hyperband/blob/master/LICENSE

from common_defs import *
import configparser as cfg
import numpy as np
#from hyperband import Hyperband
import convnet
from convnet import *
import os
from math import log, ceil
from meta_hyperband import *


def get_space(params={}):
	if not bool(params):
		space = {'init_learning_rate_space':hp.loguniform('lr', np.log(5e-5), np.log(5e-0)),#the learning rate should be from the range (5*10^-5, 5), sampled on a log-scale
		'conv1_l2_space': hp.loguniform('1l2',np.log(5e-5), np.log(5e-0)),
		'conv2_l2_space': hp.loguniform('2l2',np.log(5e-5), np.log(5e-0)),
		'conv3_l2_space': hp.loguniform('3l2', np.log(5e-5), np.log(5e-0)),
		'fc4_l2_space' :  hp.loguniform('fl2',  np.log(5e-3), np.log(5e+2)),
		'scale_space' : hp.loguniform('sc',  np.log(5e-6), np.log(5e-0)),
		'poww_space' : hp.uniform('po', 0.01, 3), 
		'learning_rate_reduction_space' :  hp.randint( 'lrr', 3)}#generates a random in  0-3 range
		return space
	if bool(params):
		alfa_center = float(params['init_learning_rate'])
		conv1_center = float(params['conv1_l2'])
		conv2_center = float(params['conv2_l2'])
		conv3_center = float(params['conv3_l2'])
		fc4_center = float(params['fc4_l2'])
		scale_center = float(params['scale'])
		poww_center = float(params['poww'])
		if alfa_center*0.8<0.00005:
			alfa_center_new_low = 0.00005
		else:
			alfa_center_new_low = alfa_center
		if alfa_center*1.2>5:
			alfa_center_new_high = 5
		else:
			alfa_center_new_high = alfa_center*1.2
		if conv1_center*0.8<0.00005:
			conv1_center_new_low = 0.00005
		else:
			conv1_center_new_low = conv1_center*0.8
		if conv1_center*1.2>5:
			conv1_center_new_high = 5
		else:
			conv1_center_new_high = conv1_center*1.2
		
		if conv2_center*0.8<0.00005:
                        conv2_center_new_low = 0.00005
                else:
                        conv2_center_new_low = conv2_center*0.8

                if conv2_center*1.2>5:
                        conv2_center_new_high = 5
                else:
                        conv2_center_new_high = conv2_center*1.2
                if conv3_center*0.8<0.00005:
                        conv3_center_new_low = 0.00005
                else:
                        conv3_center_new_low = conv3_center*0.8

                if conv3_center*1.2>5:
                        conv3_center_new_high = 5
                else:
                        conv3_center_new_high = conv3_center*1.2

		if fc4_center*0.8<0.005:
                        fc4_center_new_low = 0.005
                else:
                        fc4_center_new_low = fc4_center*0.8

                if fc4_center*1.2>500:
                        fc4_center_new_high = 500
                else:
                        fc4_center_new_high = fc4_center*1.2
		
                if scale_center*0.8<0.000005:
                        scale_center_new_low = 0.000005
                else:
                        scale_center_new_low = scale_center*0.8
		if scale_center*1.2>5:
                        scale_center_new_high = 5
                else:
                        scale_center_new_high = scale_center*1.2

                if poww_center*0.8<0.01:
                        poww_center_new_low = 0.01
                else:
                        poww_center_new_low = poww_center*0.8
                if poww_center*1.2>3:
                        poww_center_new_high = 3
                else:
                        poww_center_new_high = poww_center*1.2
		
	
		space ={
                'init_learning_rate_space':hp.loguniform('lr', np.log(alfa_center_new_low), np.log(alfa_center_new_high)),#the learning rate should be from the range (5*10^-5, 5), sampled on a log-scale
                'conv1_l2_space':hp.loguniform('1l2',np.log(conv1_center_new_low),np.log(conv1_center_new_high)),
                'conv2_l2_space':hp.loguniform('2l2',np.log(conv2_center_new_low),np.log(conv2_center_new_high)),
                'conv3_l2_space':hp.loguniform('3l2', np.log(conv3_center_new_low), np.log(conv3_center_new_high)),
                'fc4_l2_space':hp.loguniform('fl2',  np.log(fc4_center_new_low), np.log(fc4_center_new_high)),
                'scale_space':hp.loguniform('sc',  np.log(scale_center_new_low), np.log(scale_center_new_high)),
                'poww_space':hp.uniform('po', poww_center_new_low, poww_center_new_high), #as it's on linear scale I decieded to use uniform. check again to ensure
                'learning_rate_reduction_space':hp.randint( 'lrr', 3),
		'coarse_tag': params['tag']}#generates a random in  0-3 range,
		return space
		
def get_params(space,s,j):
	params={}
	params['init_learning_rate']  = sample(space['init_learning_rate_space'])
	params['conv1_l2'] = sample(space['conv1_l2_space'])
	params['conv2_l2'] = sample(space['conv2_l2_space'])
	params['conv3_l2'] = sample(space['conv3_l2_space'])
	params['fc4_l2'] = sample(space['fc4_l2_space'])
	params['scale'] = sample(space['scale_space'])
	print  params['scale']
	params['poww'] = sample(space['poww_space'])
	params['learning_rate_reduction'] = sample(space['learning_rate_reduction_space'])
	params['epsW'] = "dexp[base="+str(params['init_learning_rate'])+";tgtFactor=10"+";numSteps="+str(params['learning_rate_reduction'])+"]"
	params['step']=str(s) 
	params['tag']=str(s)+'_'+str(j)
	params['param_number'] = str(j)
	if 'coarse_tag' in space:
		params['coarse_tag'] = space['coarse_tag']	
	# if you want to enable reproduction of random configs uncomment following lines
	return handle_integers(params)
def get_meta_params(section,s,j,dataset_name,save_path):

	config = cfg.ConfigParser(inline_comment_prefixes=(';',))
        config.read('pool_for_CNN.cfg')
	while True:
		if not config.has_section(str(section)):
			space = get_space()
			return get_params(space,s,j),section+1,"random config"
		else:
			if not dataset_name in config.get(str(section),'name'):
				params = {}
				params['init_learning_rate'] = config.get(str(section), 'init_learning_rate')
				params['conv1_l2'] = config.get(str(section),'conv1_l2')
				params['conv2_l2'] = config.get(str(section),'conv2_l2')
				params['conv3_l2'] = config.get(str(section),'conv3_l2')
				params['fc4_l2'] = config.get(str(section),'fc4_l2')
				params['scale'] = config.get(str(section),'scale')
				params['poww'] = config.get(str(section),'poww')
				params['learning_rate_reduction'] = config.get(str(section),'learning_rate_reduction')
				params['epsW'] = config.get(str(section),'epsW')
				params['name_dataset'] = config.get(str(section),'name')
				params['step']=str(s)
				params['tag']=str(s)+'_'+ str(section)
				params['param_number'] = str(section)
				return handle_integers(params),section+1, config.get(str(section),'name')
			else:
				section+=1

def try_params(step_i,n_resources,params,save_path):
	config = cfg.ConfigParser(inline_comment_prefixes=(';',))
        config.read('./layers/layer-params-18pct.cfg')
        config.set('conv1', 'epsW',str(params['epsW']))
        config.set('conv2', 'epsW',str(params['epsW']))
        config.set('conv3', 'epsW',str(params['epsW']))
        config.set('fc10', 'epsW',str(params['epsW']))
        config.set('conv1', 'wc', str(params['conv1_l2']))
        config.set('conv2', 'wc', str(params['conv2_l2']))
        config.set('conv3', 'wc', str(params['conv3_l2']))
        config.set('fc10', 'wc', str(params['fc4_l2']))
        config.set('rnorm1', 'scale', str(params['scale']))
        config.set('rnorm2', 'scale', str(params['scale']))
        config.set('rnorm1', 'pow', str(params['poww']))
        config.set('rnorm2', 'pow', str(params['poww']))
        with open('./layers/layer-params-18pct.cfg', 'w') as configfile:
            config.write(configfile)
	validation_frequency = 12
	
	os.system("python convnet.py --data-provider svhn --test-range 13  --test-freq " + str(validation_frequency)+ " --train-range 1-12 --data-path /data/samin/cuda-convnet2/datasets/svhn/svhn_batches  --save-path ./svhn_checkpoints_meta31/"+str(params['step'])+"/"+str(step_i)+"/"+str(params['tag'])+" --gpu 3  --layer-def layers/layers-18pct.cfg --layer-params layers/layer-params-18pct.cfg  --inner-size 32  --mini 100 --resources " + str(n_resources))
	j = params['param_number'] 
	#to save the params
	config2 = cfg.ConfigParser()
        config2.read(save_path)
	if not config2.has_section(j):
		config2.add_section(j)
		config2.set(str(j),'init_learning_rate', str(params['init_learning_rate']))
		config2.set(str(j),'conv1_l2', str(params['conv1_l2']))
		config2.set(str(j),'conv2_l2', str(params['conv2_l2']))
		config2.set(str(j),'conv3_l2', str(params['conv3_l2']))
		config2.set(str(j),'fc4_l2', str(params['fc4_l2']))
		config2.set(str(j),'scale', str(params['scale']))
		config2.set(str(j),'poww', str(params['poww']))
		config2.set(str(j),'learning_rate_reduction', str(params['learning_rate_reduction']))
		config2.set(str(j),'epsW', str(params['epsW']))
		config2.set(str(j),'step', str(params['step']))
		config2.set(str(j),'tag', str(params['tag']))
		if 'coarse_tag' in params:
			config2.set(str(j),'coarse_tag', str(params['coarse_tag']))
		if 'name_dataset' in params:
			config2.set(str(j),'name', str(params['name_dataset']))
		else:
			config2.set(str(j),'name', 'random parameter')
	with open('resultdata.json', 'r') as fp:
		result = json.load(fp)
		loss_identtifier = 'loss'+str(step_i)
		config2.set(str(j),loss_identtifier, str(result))
	with open(save_path,'w') as cfgfile2:
		config2.write(cfgfile2)
def save_results(result, name_dataset,num_dataset):
	config = cfg.ConfigParser()
	config.read('pool_for_CNN.cfg')
	config.add_section(str(num_dataset))
	config.set(str(num_dataset),'name', str(name_dataset))
	config.set(str(num_dataset),'init_learning_rate', str(result['params']['init_learning_rate']))
	config.set(str(num_dataset),'conv1_l2', str(result['params']['conv1_l2']))
	config.set(str(num_dataset),'conv2_l2', str(result['params']['conv2_l2']))
	config.set(str(num_dataset),'conv3_l2', str(result['params']['conv3_l2']))
	config.set(str(num_dataset),'fc4_l2', str(result['params']['fc4_l2']))
	config.set(str(num_dataset),'scale', str(result['params']['scale']))
	config.set(str(num_dataset),'poww', str(result['params']['poww']))
	config.set(str(num_dataset),'learning_rate_reduction', str(result['params']['learning_rate_reduction']))
	config.set(str(num_dataset),'epsW', str(result['params']['epsW']))
	config.set(str(num_dataset),'Hyperband_best_loss', str(result['loss']))
	config.set(str(num_dataset),'step',str(result['params']['step']))
	config.set(str(num_dataset),'tag',str(result['params']['tag']))
	if 'coarse_tag' in result['params']:
		config.set(str(num_dataset),'coarse_tag',str(result['params']['coarse_tag']))
	with open("pool_for_CNN.cfg",'w') as cfgfile:
	    config.write(cfgfile)
if __name__ == "__main__":
	hb = Meta_Hyperband(get_space, get_params, get_meta_params,try_params,'svhn')
	best_result= hb.run( skip_last = 1)
	print "End of code---------------------------------------"
	print best_result
	save_results(best_result, 'svhn_meta', 31)
