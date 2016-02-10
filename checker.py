# IMP: Run as: python checker.py answer.txt submission.txt

# Ignore all warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import sys
from sklearn.metrics import classification_report as cr
original = [line.strip() for line in open(sys.argv[1], 'r')]
submission = [line.strip() for line in open(sys.argv[2], 'r')]
# target_names = ['normal', 'dos', 'r2l', 'u2r', 'probing']
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
# print x
x = x.split()
i = x.index('type5')
print x[i + 2]