import os
import argparse
from datapath import loc

parser = argparse.ArgumentParser()
parser.add_argument('--meta_tasks', type=str, default='sc,pa,po,qa,tc')
parser.add_argument('--load', type=str, default='saved')
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()
print (args)

command = 'CUDA_VISIBLE_DEVICES={:d} python3 baseline.py --task {:s} --load {:s} --lr {:s} --epochs {:s} --save {:s} --model_name {:s}'

task_types = args.meta_tasks.split(',')
list_of_tasks = []

for tt in loc['train'].keys():
	if tt[:2] in task_types:
		list_of_tasks.append(tt)

num_tasks = len(list_of_tasks)
print (list_of_tasks)

command_list = []

saved_models = os.listdir(args.load)

for tt in list_of_tasks:
	if 'qa' in tt:# and 'model_qa.pt' in saved_models:
		command_list.append(command.format(args.gpu,tt,os.path.join(args.load,'model_qa.pt'),'3e-6','2',args.load,'model_qa_ft_' + tt + '_1.pt'))
		command_list.append(command.format(args.gpu,tt,os.path.join(args.load,'model_qa.pt'),'3e-5','2',args.load,'model_qa_ft_' + tt + '_2.pt'))
	elif 'tc' in tt:# and 'model_tc.pt' in saved_models:
		command_list.append(command.format(args.gpu,tt,os.path.join(args.load,'model_tc.pt'),'3e-6','10',args.load,'model_tc_ft_' + tt + '.pt'))
# 		command_list.append(command.format(args.gpu,tt,os.path.join(args.load,'model_tc.pt'),'3e-5','10',args.load,'model_tc_ft_' + tt + '_2.pt'))
	elif 'sc' in tt:# and 'model_sc.pt' in saved_models:
		command_list.append(command.format(args.gpu,tt,os.path.join(args.load,'model_sc.pt'),'3e-6','5',args.load,'model_sc_ft_' + tt + '_1.pt'))
# 		command_list.append(command.format(args.gpu,tt,os.path.join(args.load,'model_sc.pt'),'3e-5','5',args.load,'model_sc_ft_' + tt + '_2.pt'))
	elif 'pa' in tt:# and 'model_pa.pt' in saved_models:
		command_list.append(command.format(args.gpu,tt,os.path.join(args.load,'model_pa.pt'),'3e-6','5',args.load,'model_pa_ft_' + tt + '_1.pt'))
# 		command_list.append(command.format(args.gpu,tt,os.path.join(args.load,'model_pa.pt'),'3e-5','5',args.load,'model_pa_ft_' + tt + '_2.pt'))
	elif 'po' in tt:# and 'model_po.pt' in saved_models:
		command_list.append(command.format(args.gpu,tt,os.path.join(args.load,'model_po.pt'),'3e-6','10',args.load,'model_po_ft_' + tt + '.pt'))
# 		command_list.append(command.format(args.gpu,tt,os.path.join(args.load,'model_po.pt'),'3e-5','10',args.load,'model_po_ft_' + tt + '_2.pt'))

# print (command_list)
for cmd in command_list:
	print (cmd)
	
eval_list = []
os.system('touch {:s}'.format(os.path.join(args.load,'eval_results.txt')))
eval_command = 'CUDA_VISIBLE_DEVICES={:d} python3 eval.py --meta_task {:s} --load {:s} >>{:s}'

for ft_models in os.listdir(args.load):
	if ft_models.count('.pt') > 0 and ft_models.count('ft') > 0:
		task = ft_models.split('.')[0].split('_')[3]
		eval_list.append(eval_command.format(args.gpu,task,os.path.join(args.load,ft_models),os.path.join(args.load,'eval_results.txt')))

# for cmd in eval_list:
# 	print (cmd)
