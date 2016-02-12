from sklearn.metrics import classification_report as cr

def check(input_file, gold_file):

	original = [line.strip() for line in open(gold_file, 'r')]
	submission = [line.strip() for line in open(input_file, 'r')]
	target_names = ['normal', 'dos', 'r2l', 'u2r', 'probing']
	target_names = ['type1', 'type2', 'type4', 'type5', 'type3']

	for i in range(len(submission)):
		if submission[i] not in target_names:
			submission[i] = ''
	if len(submission) < len(original):
		extra = ['' for x in range(len(original) - len(submission))]
		submission += extra
	elif len(submission) > len(original):
		submission = submission[0:len(original)]
	x = cr(original, submission, digits = 4)
	x = x.split()
	ind5 = x.index('type5')
	score = x[ind5 + 2]
	j = x.index('avg')
	score = x[ind5 + 2]
	precision = x[j + 3]
	if (float(precision) > 0 and float(score) > 0):
		return 0,0,(float(score)+float(precision))*50, "Accepted" 
	else:
		return 0,0,0, "Wrong Answer"
