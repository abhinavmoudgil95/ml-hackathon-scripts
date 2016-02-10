# IMP: Run as: python checker.py answer.txt submission.txt

# Ignore all warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import sys
from sklearn.metrics import classification_report as cr
original = [line.strip() for line in open(sys.argv[0], 'r')]
submission = [line.strip() for line in open(sys.argv[1], 'r')]
target_names = ['Normal', 'DOS', 'R2L', 'U2R']
for i in range(len(submission)):
	if submission[i] not in target_names:
		submission[i] = ''
if len(submission) < len(original):
	extra = ['' for x in range(len(original) - len(submission))]
	submission += extra
elif len(submission) > len(original):
	submission = submission[0:len(original)]
x = cr(original, submission, digits = 4).split()
i = x.index('U2R')
print x[i + 2]