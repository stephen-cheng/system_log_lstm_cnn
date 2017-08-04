features of each line in log_dataset:

timestamp, syslogfacility, syslogfacilityText, syslogseverity, syslogseverityText, msg

run:

log2dataset.py:      generate log txt file from mysql
dataset2label.py:    split log txt file into 128X time-step text and label
msg_tokenize.py:     convert msg text into token
msg_tokenize_reshape.py:  change the size of token and label to 128X
