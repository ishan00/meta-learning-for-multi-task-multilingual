import matplotlib
matplotlib.use('Agg')
import pickle
import matplotlib.pyplot as plt

with open('DDS_TARGET_QA_LR_2/logger.pickle','rb') as g:
	data = pickle.load(g)

lang_cnt = {}
for tk in data['val_loss'].keys():
	lang_cnt[tk[:2]] = lang_cnt.get(tk[:2],0) + 1

psis = {k:[] for k in data['val_loss'].keys()}

psis_avg = {
	'qa':[0 for _ in range(len(data['psis']))],
	'sc':[0 for _ in range(len(data['psis']))],
	'tc':[0 for _ in range(len(data['psis']))],
	'po':[0 for _ in range(len(data['psis']))],
	'pa':[0 for _ in range(len(data['psis']))],
}

for i,row in enumerate(data['psis']):
	for elem in row:
		elem = elem.split(',')
		psis[elem[0]].append(float(elem[1]))
		psis_avg[elem[0][:2]][i] += float(elem[1]) / lang_cnt[elem[0][:2]]

# for task in psis_avg.keys():
# 	plt.plot([10*i + 10 for i in range(0,len(psis_avg[task]),1)],[psis_avg[task][i] for i in range(0,len(psis_avg[task]),1)],label=task)

for task in psis.keys():
	if 'qa' in task:
		plt.plot([10*i + 10 for i in range(0,len(psis[task]),1)],[psis[task][i] for i in range(0,len(psis[task]),1)],label=task)


		
plt.xlabel('Meta Step')
plt.ylabel('Task Samplig Probability')
plt.legend(prop={'size': 8})
plt.savefig('qa_lang_psis.jpg')
