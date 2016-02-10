from sklearn.metrics import classification_report as cr
original = [line.strip() for line in open("1.txt", 'r')]
submission = [line.strip() for line in open("2.txt", 'r')]
if len(submission) < len(original):
	extra = ['' for x in range(len(original) - len(submission))]
	submission += extra
elif len(submission) > len(original):
	submission = submission[0:len(original)]
print cr(original, submission)