#used another code as reference: Copyright (c) 2017, Zygmunt Zając
#https://github.com/zygmuntz/hyperband/blob/master/LICENSE
import numpy as np
from random import random
from math import log, ceil
from time import time, ctime
import json
import os
class Meta_Hyperband:
	def __init__( self,get_space_function, get_params_function, get_meta_params_function, try_params_function,dataset_name):
		self.get_space = get_space_function
		self.get_params = get_params_function
		self.get_meta_params = get_meta_params_function
		self.try_params = try_params_function
		self.max_iter = 600  	# maximum iterations per configuration, this can be changed according to the problem you are trying to solve and the amount of resources you would like to assign
		self.eta = 4			# defines configuration downsampling rate (debracket_indexult = 4)
		self.logeta = lambda x: log( x ) / log( self.eta )
		self.s_max = int( self.logeta( self.max_iter ))
		self.B = ( self.s_max + 1 ) * self.max_iter
		self.testingmatters = []
		self.results = []	# list of dicts
		self.counter = 0
		self.best_loss = np.inf
		self.best_counter = 0
		self.iterations = []
		self.svalues = []
		self.space = {}
		self.dataset_name = dataset_name
		self.meta_names = []
		self.best_counter42 =0 # keeps a pointer to the best of brackets 2 and 4
		self.best_counter1 = 0 # keeps a pointer to the best of bracket 1
		self.best_loss1 =  np.inf # keeps the validation loss of the best config of bracket 1
		self.best_loss42 =  np.inf # keeps the validation loss of the best config of brackets 2 and 4
#----------------------------------------------------------------------------------------------------------------------------

	def run( self, skip_last = 0, dry_run = bracket_indexlse):
		brackets_sequence = [1,4,2,3,0]
		for bracket_index in range(5):
			s = brackets_sequence[bracket_index]
			self.svalues.append(s)
			n = int(ceil( self.B / self.max_iter / ( s + 1 ) * self.eta ** s ))	
			r = self.max_iter * self.eta ** ( -s )
			path_to_saved_random_configs = "./svhn_checkpoints_meta31/"+str(s)	# the sampled configurations in different brackets for each experiment are saved in a directory corresponding to that experiment identified by the experiment number and dataset name, followed by a directory specifying the bracket index
			if not os.path.exists(path_to_saved_random_configs):
            			 os.makedirs(path_to_saved_random_configs)
        		full_path = os.path.join(path_to_saved_random_configs, str(s)+'th_config_file.cfg')
        		f = open(full_path, 'w')
			T=[]
            if s==1: #bracket 1, meta-learning
				section = 0
				j = 0
				while j<n:
					params, section,meta_name = self.get_meta_params(section,s,j,self.dataset_name,full_path)
					self.meta_names.append(meta_name)
					T.append(params)
					j+=1
			elif s==4 or s==2: # random search in brackets 2 and 4
				self.space = self.get_space()
				T = [self.get_params(self.space,s,j) for j in range( n )]
			elif s==3:#bracket 3, Coarse-to-Fine
				self.space = self.get_space(self.results[self.best_counter42]['params'])
				for j in range(n/2):
					T.append(self.get_params(self.space,s,j))
				self.space = self.get_space(self.results[self.best_counter1]['params'])
				for j in range(n/2,n):
					 T.append(self.get_params(self.space,s,j))	
			elif s==0: #bracket 0, Coarse-to-Fine
				self.space = self.get_space(self.results[self.best_counter1]['params'])
				T = [ self.get_params(self.space,s,j) for j in range( n )]
			
			
            #Successive Halving inside each bracket
			for i in range(s+1):
					
				self.iterations.append(i)
				n_configs = n * self.eta ** ( -i )
				n_iterations = int(round(r * self.eta ** ( i )))
				self.testingmatters.append ({'s':s,'i':i, 'r':n_iterations, 'n':n_configs})
				print "\n*** {} configurations x {:.1f} resources (batches) each".format( 
						n_configs, n_iterations )
				val_losses = []
				early_stops = []
				for t in T:
					print "\n{} | {} | lowest loss so bracket_indexr: {:.4f} (run {})\n".format( 
							self.counter, ctime(), self.best_loss, self.best_counter )
					start_time = time()
					self.try_params(i, n_iterations, t,full_path)		
					with open('resultdata.json', 'r') as fp:
						result = json.load(fp)
					seconds = int( round( time() - start_time ))
					print "\n{} seconds.".format( seconds )
					loss = result['loss']	
					early_stop = result['Early_stop']
					val_losses.append( loss )
					early_stops.append( early_stop )
				
					if loss < self.best_loss:
						self.best_loss = loss
						self.best_counter = self.counter
					if s==1 and loss<self.best_loss1:
						self.best_loss1 = loss
						self.best_counter1 = self.counter
					if s==4 or s==2  and loss<self.best_loss42:
						self.best_loss42 = loss
                                                self.best_counter42 = self.counter
					result['counter'] = self.counter
					result['seconds'] = seconds
					result['params'] = t
					result['iterations'] = n_iterations						
					self.results.append( result )
					self.counter += 1
				indices = np.argsort( val_losses )
				T = [ T[i] for i in indices if not early_stops[i]]
				T = T[ 0:int( n_configs / self.eta )]
		return  self.results[self.best_counter]
